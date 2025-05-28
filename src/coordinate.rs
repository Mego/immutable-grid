use std::ops::Index;

use generic_array::*;

use crate::grid::ImmutableGrid;

pub struct Coordinate {
    row: usize,
    col: usize,
}

impl From<(usize, usize)> for Coordinate {
    fn from((row, col): (usize, usize)) -> Self {
        Coordinate { row, col }
    }
}

impl From<Coordinate> for (usize, usize) {
    fn from(value: Coordinate) -> Self {
        (value.row, value.col)
    }
}

impl<T, N: ArrayLength> Index<Coordinate> for ImmutableGrid<T, N>
where
    T: Clone + Copy,
    GenericArray<T, N>: Copy,
{
    type Output = T;

    fn index(&self, index: Coordinate) -> &Self::Output {
        self.get(index.row, index.col).unwrap()
    }
}
