use std::{fmt, hash, iter::StepBy, ops::Add, slice::Iter};

use generic_array::{
    ArrayLength, GenericArray, IntoArrayLength,
    functional::FunctionalSequence,
    sequence::GenericSequence,
    typenum::{Const, Sum},
};

use crate::{
    coordinate::Coordinate,
    iter::{GridColIter, GridRowIter},
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ImmutableGrid<T, N: ArrayLength>
where
    T: Clone + Copy,
    GenericArray<T, N>: Copy,
{
    items: GenericArray<T, N>,
    rows: usize,
    cols: usize,
}

impl<T, N: ArrayLength> ImmutableGrid<T, N>
where
    T: Clone + Copy,
    GenericArray<T, N>: Copy,
{
    pub fn new(cols: usize) -> Self
    where
        T: Default,
    {
        Self::init(cols, T::default())
    }

    pub fn init(cols: usize, value: T) -> Self {
        assert!(cols > 0, "grid must not be empty");
        assert_eq!(
            N::to_usize() % cols,
            0,
            "total length ({}) must be divisible by number of columns ({cols})",
            N::to_usize()
        );
        Self {
            items: GenericArray::generate(|_| value),
            cols,
            rows: N::to_usize() / cols,
        }
    }

    pub fn from_array<const U: usize>(array: [T; U], cols: usize) -> Self
    where
        Const<U>: IntoArrayLength<ArrayLength = N>,
    {
        assert!(cols > 0, "grid must not be empty");
        assert_eq!(
            N::to_usize() % cols,
            0,
            "total length ({}) must be divisible by number of columns ({cols})",
            N::to_usize()
        );
        Self {
            items: GenericArray::from_array(array),
            cols,
            rows: N::to_usize() / cols,
        }
    }

    pub fn from_generic_array(array: GenericArray<T, N>, cols: usize) -> Self {
        assert!(cols > 0, "grid must not be empty");
        assert_eq!(
            N::to_usize() % cols,
            0,
            "total length ({}) must be divisible by number of columns ({cols})",
            N::to_usize()
        );
        Self {
            items: array,
            cols,
            rows: N::to_usize() / cols,
        }
    }

    const fn get_index(&self, row: usize, col: usize) -> usize {
        row * self.cols() + col
    }

    fn get_into_index(&self, idx: impl Into<(usize, usize)>) -> usize {
        let i = idx.into();
        self.get_index(i.0, i.1)
    }

    pub fn get(&self, row: impl TryInto<usize>, col: impl TryInto<usize>) -> Option<&T> {
        let row_usize = row.try_into().ok()?;
        let col_usize = col.try_into().ok()?;
        if col_usize < self.cols {
            let index = self.get_index(row_usize, col_usize);
            if index < N::to_usize() {
                return Some(&self.items[index]);
            }
        }
        None
    }

    pub fn get_coord(&self, coord: Coordinate) -> Option<&T> {
        self.get(coord.row, coord.col)
    }

    pub const fn size(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    pub const fn len(&self) -> usize {
        self.rows() * self.cols()
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn cols(&self) -> usize {
        self.cols
    }

    pub fn iter(&self) -> Iter<T> {
        self.items.iter()
    }

    pub fn iter_col(&self, col: usize) -> StepBy<Iter<'_, T>> {
        assert!(col < self.cols());
        self.items[col..].iter().step_by(self.cols)
    }

    pub fn iter_row(&self, row: usize) -> Iter<'_, T> {
        assert!(row < self.rows());
        let start = row * self.cols();
        self.items[start..(start + self.cols())].iter()
    }

    pub fn indexed_iter(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        self.items
            .iter()
            .enumerate()
            .map(move |(idx, i)| ((idx / self.cols, idx % self.cols), i))
    }

    pub const fn flatten(&self) -> &GenericArray<T, N> {
        &self.items
    }

    pub fn into_vec(self) -> Vec<T> {
        self.items.to_vec()
    }

    pub fn transposed(&self) -> Self {
        let mut new_items = self.items.clone();
        for (idx, item) in self.indexed_iter() {
            new_items[self.get_index(idx.1, idx.0)] = *item;
        }
        Self::from_generic_array(new_items, self.rows())
    }

    pub fn iter_rows(&self) -> GridRowIter<T, N> {
        GridRowIter {
            grid: self,
            row_start_index: 0,
            row_end_index: self.rows(),
        }
    }

    pub fn iter_cols(&self) -> GridColIter<T, N> {
        GridColIter {
            grid: self,
            col_start_index: 0,
            col_end_index: self.cols(),
        }
    }

    pub fn flipped_cols(&self) -> Self {
        let mut new_items = self.items.clone();
        for row in 0..self.rows() {
            let idx = row * self.cols();
            new_items[idx..idx + self.cols()].reverse();
        }
        Self::from_generic_array(new_items, self.cols())
    }

    pub fn flipped_rows(&self) -> Self {
        let mut new_items = self.items.clone();
        for row in 0..self.rows() / 2 {
            for col in 0..self.cols() {
                let cell1 = self.get_index(row, col);
                let cell2 = self.get_index(self.rows() - row - 1, col);
                new_items.swap(cell1, cell2);
            }
        }
        Self::from_generic_array(new_items, self.cols())
    }

    pub fn rotated_left(&self) -> Self {
        self.transposed().flipped_rows()
    }

    pub fn rotated_right(&self) -> Self {
        self.transposed().flipped_cols()
    }

    pub fn rotated_half(&self) -> Self {
        let mut new_items = self.items.clone();
        new_items.reverse();
        Self::from_generic_array(new_items, self.cols())
    }

    pub fn map<U, F>(&self, f: F) -> ImmutableGrid<U, N>
    where
        U: Clone + Copy,
        F: Fn(&T) -> U,
        GenericArray<U, N>: Copy,
    {
        ImmutableGrid::from_generic_array(self.items.map(|x| f(&x)), self.cols())
    }

    pub fn append_row<
        R: ArrayLength + Add<N>,
        U: ArrayLength + IntoArrayLength<ArrayLength = Sum<R, N>>,
    >(
        &self,
        row: &GenericArray<T, R>,
    ) -> ImmutableGrid<T, U>
    where
        GenericArray<T, U>: Copy,
    {
        ImmutableGrid::from_generic_array(
            GenericArray::from_iter(self.items.iter().cloned().chain(row.iter().cloned())),
            self.cols(),
        )
    }

    pub fn append_col<
        C: ArrayLength + Add<N>,
        U: ArrayLength + IntoArrayLength<ArrayLength = Sum<C, N>>,
    >(
        &self,
        col: &GenericArray<T, C>,
    ) -> ImmutableGrid<T, U>
    where
        GenericArray<T, U>: Copy,
    {
        ImmutableGrid::from_generic_array(
            GenericArray::from_iter(
                self.iter_rows()
                    .enumerate()
                    .map(|(i, row)| row.chain([&col[i]]))
                    .flatten()
                    .cloned(),
            ),
            self.cols() + 1,
        )
    }

    pub fn filled(&self, value: T) -> Self {
        let mut items = self.items.clone();
        items.fill(value);
        Self::from_generic_array(items, self.cols())
    }

    pub fn filled_with<F: FnMut() -> T>(&self, f: F) -> Self {
        let mut items = self.items.clone();
        items.fill_with(f);
        Self::from_generic_array(items, self.cols())
    }

    pub fn swapped(&self, a: impl Into<(usize, usize)>, b: impl Into<(usize, usize)>) -> Self {
        let a_idx = self.get_into_index(a);
        let b_idx = self.get_into_index(b);
        Self::from_generic_array(
            GenericArray::from_iter(self.items.iter().enumerate().map(|(i, val)| {
                if i == a_idx {
                    self.items[b_idx]
                } else if i == b_idx {
                    self.items[a_idx]
                } else {
                    *val
                }
            })),
            self.cols(),
        )
    }
}

impl<T, N: ArrayLength> Default for ImmutableGrid<T, N>
where
    T: Clone + Copy + Default,
    GenericArray<T, N>: Copy,
{
    fn default() -> Self {
        Self::new(1)
    }
}

// copied from the grid crate, with slight modifications
impl<T, N: ArrayLength> fmt::Debug for ImmutableGrid<T, N>
where
    T: Clone + Copy + fmt::Debug,
    GenericArray<T, N>: Copy,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        if self.cols > 0 {
            if f.alternate() {
                writeln!(f)?;
                /*
                    WARNING

                    Compound types becoming enormous as the entire `fmt::Debug` width is applied to each item individually.
                    For tuples and structs define padding and precision arguments manually to improve readability.
                */
                let width = f.width().unwrap_or_else(|| {
                    // Conditionally calculate the longest item by default.
                    self.items
                        .iter()
                        .map(|i| format!("{i:?}").len())
                        .max()
                        .unwrap()
                });
                let precision = f.precision().unwrap_or(2);
                for mut row in self.iter_rows().map(Iterator::peekable) {
                    write!(f, "    [")?;
                    while let Some(item) = row.next() {
                        write!(f, " {item:width$.precision$?}")?;
                        if row.peek().is_some() {
                            write!(f, ",")?;
                        }
                    }
                    writeln!(f, "]")?;
                }
            } else {
                for row in self.iter_rows() {
                    f.debug_list().entries(row).finish()?;
                }
            }
        }
        write!(f, "]")
    }
}

impl<T, N: ArrayLength, const U: usize> From<([T; U], usize)> for ImmutableGrid<T, N>
where
    T: Clone + Copy,
    GenericArray<T, N>: Copy,
    Const<U>: IntoArrayLength<ArrayLength = N>,
{
    fn from((value, cols): ([T; U], usize)) -> Self {
        Self::from_array(value, cols)
    }
}

impl<T, N: ArrayLength, const U: usize> From<(&[T; U], usize)> for ImmutableGrid<T, N>
where
    T: Clone + Copy,
    GenericArray<T, N>: Copy,
    Const<U>: IntoArrayLength<ArrayLength = N>,
{
    fn from((value, cols): (&[T; U], usize)) -> Self {
        Self::from_array(*value, cols)
    }
}

impl<T, N: ArrayLength> hash::Hash for ImmutableGrid<T, N>
where
    T: Clone + Copy + hash::Hash,
    GenericArray<T, N>: Copy,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.rows.hash(state);
        self.cols.hash(state);
        self.items.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use generic_array::typenum;

    use super::*;

    #[test]
    fn test_init() {
        let g = ImmutableGrid::<f64, typenum::U4>::init(2, 1.0);
        assert_eq!(g.cols(), 2);
        assert_eq!(g.rows(), 2);
        assert_eq!(g.items.into_array(), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    #[should_panic]
    fn test_init_not_square() {
        ImmutableGrid::<f64, typenum::U5>::init(2, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_init_empty() {
        ImmutableGrid::<f64, typenum::U0>::init(0, 1.0);
    }

    #[test]
    fn test_from_array() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        assert_eq!(g.items.into_array(), arr);
    }

    #[test]
    fn test_from_generic_array() {
        let arr = GenericArray::from_array([1, 2, 3, 4]);
        let g = ImmutableGrid::from_generic_array(arr, 2);
        assert_eq!(g.items, arr);
    }

    #[test]
    fn test_iter_col() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let items: Vec<_> = g.iter_col(0).collect();
        assert_eq!(items, vec![&1, &3]);
    }

    #[test]
    fn test_iter_row() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let items: Vec<_> = g.iter_row(0).collect();
        assert_eq!(items, vec![&1, &2]);
    }

    #[test]
    fn test_indexed_iter() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let items: Vec<_> = g.indexed_iter().collect();
        assert_eq!(
            items,
            vec![((0, 0), &1), ((0, 1), &2), ((1, 0), &3), ((1, 1), &4)]
        );
    }

    #[test]
    fn test_transposed() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let g2 = g.transposed();
        assert_eq!(g2.items.into_array(), [1, 3, 2, 4]);
    }

    #[test]
    fn test_iter_rows() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let items: Vec<Vec<_>> = g.iter_rows().map(|i| i.collect()).collect();
        assert_eq!(items, vec![vec![&1, &2], vec![&3, &4]]);
    }

    #[test]
    fn test_iter_cols() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let items: Vec<Vec<_>> = g.iter_cols().map(|i| i.collect()).collect();
        assert_eq!(items, vec![vec![&1, &3], vec![&2, &4]]);
    }

    #[test]
    fn test_flipped_rows() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let g2 = g.flipped_rows();
        assert_eq!(g2.items.into_array(), [3, 4, 1, 2]);
    }

    #[test]
    fn test_flipped_cols() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let g2 = g.flipped_cols();
        assert_eq!(g2.items.into_array(), [2, 1, 4, 3]);
    }

    #[test]
    fn test_rotated_left() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let g2 = g.rotated_left();
        assert_eq!(g2.items.into_array(), [2, 4, 1, 3]);
    }

    #[test]
    fn test_rotated_right() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let g2 = g.rotated_right();
        assert_eq!(g2.items.into_array(), [3, 1, 4, 2]);
    }

    #[test]
    fn test_rotated_half() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let g2 = g.rotated_half();
        assert_eq!(g2.items.into_array(), [4, 3, 2, 1]);
    }

    #[test]
    fn test_map() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let g2 = g.map(|x| x + 2);
        assert_eq!(g2.items.into_array(), [3, 4, 5, 6]);
    }

    #[test]
    fn test_append_row() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let g2 = g.append_row(&GenericArray::from_array([5, 6]));
        assert_eq!(g2.items.into_array(), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_append_col() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let g2 = g.append_col(&GenericArray::from_array([5, 6]));
        assert_eq!(g2.items.into_array(), [1, 2, 5, 3, 4, 6]);
    }

    #[test]
    fn test_filled() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let g2 = g.filled(0);
        assert_eq!(g2.items.into_array(), [0, 0, 0, 0]);
    }

    #[test]
    fn test_filled_with() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let g2 = g.filled_with(usize::default);
        assert_eq!(g2.items.into_array(), [0, 0, 0, 0]);
    }

    #[test]
    fn test_swapped() {
        let arr = [1, 2, 3, 4];
        let g = ImmutableGrid::from_array(arr, 2);
        let g2 = g.swapped((0, 0), (1, 1));
        assert_eq!(g2.items.into_array(), [4, 2, 3, 1]);
    }
}
