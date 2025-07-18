use generic_array::ArrayLength;

use crate::{coordinate::Coordinate, grid::ImmutableGrid};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Right,
    Up,
    Left,
    Down,
}

impl Direction {
    pub const fn rotate_left(self) -> Self {
        match self {
            Direction::Down => Direction::Right,
            Direction::Left => Direction::Down,
            Direction::Right => Direction::Up,
            Direction::Up => Direction::Left,
        }
    }

    pub const fn rotate_right(self) -> Self {
        match self {
            Direction::Down => Direction::Left,
            Direction::Left => Direction::Up,
            Direction::Right => Direction::Down,
            Direction::Up => Direction::Right,
        }
    }

    pub const fn rotate_half(self) -> Self {
        match self {
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
            Direction::Up => Direction::Down,
        }
    }
}

impl From<Direction> for (isize, isize) {
    fn from(value: Direction) -> Self {
        match value {
            Direction::Right => (1, 0),
            Direction::Up => (0, -1),
            Direction::Left => (-1, 0),
            Direction::Down => (0, 1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub coordinate: Coordinate,
    pub direction: Direction,
}

impl Position {
    pub fn is_valid<T: Clone + Copy, N: ArrayLength>(&self, grid: &ImmutableGrid<T, N>) -> bool
    where
        N::ArrayType<T>: Copy,
    {
        grid.get_coord(self.coordinate).is_some()
    }

    pub const fn forward(self) -> Self {
        let coordinate = match self.direction {
            Direction::Down => self.coordinate.down(),
            Direction::Left => self.coordinate.left(),
            Direction::Right => self.coordinate.right(),
            Direction::Up => self.coordinate.up(),
        };
        Self { coordinate, ..self }
    }

    pub const fn rotate_left(self) -> Self {
        Self {
            direction: self.direction.rotate_left(),
            ..self
        }
    }

    pub const fn rotate_right(self) -> Self {
        Self {
            direction: self.direction.rotate_right(),
            ..self
        }
    }

    pub const fn rotate_half(self) -> Self {
        Self {
            direction: self.direction.rotate_half(),
            ..self
        }
    }

    pub const fn backward(self) -> Self {
        self.rotate_half().forward()
    }

    pub const fn left(self) -> Self {
        self.rotate_left().forward()
    }

    pub const fn right(self) -> Self {
        self.rotate_right().forward()
    }
}

impl From<(Coordinate, Direction)> for Position {
    fn from((coordinate, direction): (Coordinate, Direction)) -> Self {
        Self {
            coordinate,
            direction,
        }
    }
}

impl From<(Direction, Coordinate)> for Position {
    fn from((direction, coordinate): (Direction, Coordinate)) -> Self {
        Self {
            coordinate,
            direction,
        }
    }
}
