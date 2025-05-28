use std::{iter::StepBy, slice::Iter};

use generic_array::*;

use crate::grid::ImmutableGrid;

#[derive(Clone)]
pub struct GridRowIter<'a, T: Clone + Copy, N: ArrayLength>
where
    GenericArray<T, N>: Copy,
{
    pub(crate) grid: &'a ImmutableGrid<T, N>,
    pub(crate) row_start_index: usize,
    pub(crate) row_end_index: usize,
}

#[derive(Clone)]
pub struct GridColIter<'a, T: Clone + Copy, N: ArrayLength>
where
    GenericArray<T, N>: Copy,
{
    pub(crate) grid: &'a ImmutableGrid<T, N>,
    pub(crate) col_start_index: usize,
    pub(crate) col_end_index: usize,
}

impl<'a, T: Clone + Copy, N: ArrayLength> Iterator for GridRowIter<'a, T, N>
where
    GenericArray<T, N>: Copy,
{
    type Item = Iter<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row_start_index >= self.row_end_index {
            return None;
        }

        let row_iter = self.grid.iter_row(self.row_start_index);
        self.row_start_index += 1;
        Some(row_iter)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.row_end_index - self.row_start_index;
        (size, Some(size))
    }
}

impl<'a, T: Clone + Copy, N: ArrayLength> ExactSizeIterator for GridRowIter<'a, T, N> where
    GenericArray<T, N>: Copy
{
}

impl<'a, T: Clone + Copy, N: ArrayLength> DoubleEndedIterator for GridRowIter<'a, T, N>
where
    GenericArray<T, N>: Copy,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.row_start_index >= self.row_end_index {
            return None;
        }

        let row_iter = self.grid.iter_row(self.row_end_index - 1);
        self.row_end_index -= 1;
        Some(row_iter)
    }
}

impl<'a, T: Clone + Copy, N: ArrayLength> Iterator for GridColIter<'a, T, N>
where
    GenericArray<T, N>: Copy,
{
    type Item = StepBy<Iter<'a, T>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.col_start_index >= self.col_end_index {
            return None;
        }

        let col_iter = self.grid.iter_col(self.col_start_index);
        self.col_start_index += 1;
        Some(col_iter)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.col_end_index - self.col_start_index;
        (size, Some(size))
    }
}

impl<'a, T: Clone + Copy, N: ArrayLength> ExactSizeIterator for GridColIter<'a, T, N> where
    GenericArray<T, N>: Copy
{
}

impl<'a, T: Clone + Copy, N: ArrayLength> DoubleEndedIterator for GridColIter<'a, T, N>
where
    GenericArray<T, N>: Copy,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.col_start_index >= self.col_end_index {
            return None;
        }

        let col_iter = self.grid.iter_col(self.col_end_index - 1);
        self.col_end_index -= 1;
        Some(col_iter)
    }
}
