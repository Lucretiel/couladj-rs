use std::{collections::HashSet, path::PathBuf, time::Instant};

use anyhow::Context;
use cascade::cascade;
use gridly::prelude::*;
use image::{io::Reader as ImageReader, GenericImageView, Rgba};
use mimalloc::MiMalloc;
use rayon::prelude::*;
use structopt::StructOpt;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn to_index(location: Location, dimensions: Vector) -> usize {
    (location.row.0 as usize * dimensions.columns.0 as usize) + (location.column.0 as usize)
}

fn from_index(index: usize, dimensions: Vector) -> Location {
    let row = Row((index as isize).div_euclid(dimensions.columns.0));
    let column = Column((index as isize).rem_euclid(dimensions.columns.0));

    row + column
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PixelPair {
    origin: Rgba<u8>,
    neighbor: Rgba<u8>,
}

impl PixelPair {
    fn swap(&self) -> Self {
        Self {
            origin: self.neighbor,
            neighbor: self.origin,
        }
    }
}

impl PartialOrd for PixelPair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PixelPair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.origin
            .0
            .cmp(&other.origin.0)
            .then_with(|| self.neighbor.0.cmp(&other.neighbor.0))
    }
}

#[derive(Debug, Clone, Copy)]
struct Rectangle {
    dimensions: Vector,
}

impl GridBounds for Rectangle {
    fn dimensions(&self) -> Vector {
        self.dimensions
    }

    fn root(&self) -> Location {
        Location::zero()
    }
}

/// Run the couladj algorithm, using rayon for multithreading.
///
/// `buffer` is a 2D buffer of pixels, flattened in row- or column- major order.
/// `dimensions` is the dimensions of the original image
/// `adjacencies` is the directions that are checked per-pixel
#[inline]
fn couladj_generic_rayon<A>(
    buffer: &[Rgba<u8>],
    dimensions: Vector,
    adjacencies: A,
) -> HashSet<PixelPair>
where
    A: Iterator<Item = Vector> + Clone + Sync,
{
    let rect = Rectangle { dimensions };
    buffer
        .par_iter()
        .copied()
        .enumerate()
        .flat_map_iter(|(index, pixel)| {
            let location = from_index(index, rect.dimensions);
            adjacencies
                .clone()
                .map(move |neighbor_vec| location + neighbor_vec)
                .filter_map(|neighbor| rect.check_location(neighbor).ok())
                .map(|neighbor| to_index(neighbor, rect.dimensions))
                .map(|neighbor| buffer[neighbor])
                .filter(move |&neighbor| neighbor != pixel)
                .map(move |neighbor| PixelPair {
                    origin: pixel,
                    neighbor,
                })
        })
        .fold(HashSet::new, |set, pair| cascade!(set; ..insert(pair);))
        .reduce(HashSet::new, |set1, set2| {
            match set1.capacity() > set2.capacity() {
                true => cascade!(set1; ..extend(set2);),
                false => cascade!(set2; ..extend(set1);),
            }
        })
}

#[derive(Debug, StructOpt)]
struct Args {
    #[structopt(short, long, help = "The image file to analyze")]
    file: PathBuf,

    #[structopt(
        short = "a",
        long,
        help = "If given, adjacencies will be computed for all 8 directions, rather than the 4 cardinal directions"
    )]
    full_adjacencies: bool,

    #[structopt(
        short,
        long,
        help = "Instead of a tsv, just input the number of unique pairs"
    )]
    count: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::from_args();

    eprintln!("Loading image...");
    let now = Instant::now();
    let (dimensions, buffer) = {
        let image = ImageReader::open(&args.file)
            .with_context(|| format!("Failed to open image file '{:?}'", args.file))?
            .decode()
            .with_context(|| format!("Failed to read image file data from '{:?}'", args.file))?;

        let pixel_buffer: Vec<Rgba<u8>> = image.pixels().map(|(_, _, pixel)| pixel).collect();

        let (x, y) = image.dimensions();
        let dimensions = Vector {
            rows: Rows(y as isize),
            columns: Columns(x as isize),
        };

        (dimensions, pixel_buffer)
    };
    eprintln!("  {:?}", now.elapsed());

    eprintln!("Calculating adjacencies...");
    let now = Instant::now();
    let mut result = match args.full_adjacencies {
        false => couladj_generic_rayon(
            &buffer,
            dimensions,
            [Down.as_vector(), Right.as_vector()].iter().copied(),
        ),
        true => couladj_generic_rayon(
            &buffer,
            dimensions,
            [
                Down.as_vector(),
                Right.as_vector(),
                Down + Left,
                Down + Right,
            ]
            .iter()
            .copied(),
        ),
    };
    eprintln!("  {:?}", now.elapsed());

    // We only search for 1-way adjacencies; make sure our set is bidirectional
    result.extend(result.clone().iter().map(|pair| pair.swap()));

    if args.count {
        println!("Found {} unique adjacencies", result.len())
    } else {
        let data = {
            let mut data: Vec<PixelPair> = result.iter().copied().collect();
            data.sort_unstable();
            data
        };

        println!("r\tg\tb\ta\tadj_r\tadj_g\tadj_b\tadj_a");

        data.iter().for_each(|pair| {
            println!(
                "{r}\t{g}\t{b}\t{a}\t{nr}\t{ng}\t{nb}\t{na}",
                r = pair.origin.0[0],
                g = pair.origin.0[1],
                b = pair.origin.0[2],
                a = pair.origin.0[3],
                nr = pair.neighbor.0[0],
                ng = pair.neighbor.0[1],
                nb = pair.neighbor.0[2],
                na = pair.neighbor.0[3],
            )
        });
    }

    Ok(())
}
