use std::ops::Index;

use generic_array::*;

use crate::grid::ImmutableGrid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coordinate {
    pub row: usize,
    pub col: usize,
}

impl Coordinate {
    pub fn is_valid<T: Clone + Copy, N: ArrayLength>(&self, grid: &ImmutableGrid<T, N>) -> bool
    where
        N::ArrayType<T>: Copy,
    {
        grid.get_coord(*self).is_some()
    }

    pub const fn up(self) -> Self {
        Self {
            row: self.row - 1,
            ..self
        }
    }

    pub const fn down(self) -> Self {
        Self {
            row: self.row + 1,
            ..self
        }
    }

    pub const fn left(self) -> Self {
        Self {
            col: self.col - 1,
            ..self
        }
    }

    pub const fn right(self) -> Self {
        Self {
            col: self.col + 1,
            ..self
        }
    }
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
