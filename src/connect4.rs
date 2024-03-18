use crate::mcts::{Node, GameState};
use std::io::stdout;
use crossterm::{
    cursor::{Show, Hide, MoveTo},
    event::{read, Event, KeyCode},
    execute,
    terminal::{self, Clear, ClearType},
    Result,
};


#[derive(Clone)]
struct State {
    board: Vec<Vec<char>>,
    player: char,
}

impl State {
    fn new() -> State {
        State {
            board: vec![vec![' '; 7]; 6],
            player: 'X',
        }
    }

    fn get_top_row(&self, col: usize) -> Option<usize> {
        self.board.iter().position(|row| row[col] == ' ')
    }
}

impl GameState for State {
    type Action = usize;

    fn is_terminal(&self) -> bool {
        self.get_result() != 0 || self.board.iter().all(|row| row.iter().all(|&cell| cell != ' '))
    }

    fn get_player_turn(&self) -> i32 {
        match self.player {
            'X' => 1,
            'O' => -1,
            _ => unreachable!(),
        }
    }

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        (0..self.board[0].len())
            .filter(|&col| self.board[5][col] == ' ')
            .collect()
    }

    fn get_policy(&self, actions: &Vec<Self::Action>) -> Vec<f64> {
        let mut scores = Vec::new();
        for &action in actions {
            let mut next_state = self.get_next_state(action);
            let mut score = 100.0;
            if next_state.get_result() != 0 { score += 300.0; }
            next_state = self.clone();
            next_state.player = if self.player == 'X' { 'O' } else { 'X' };
            next_state = next_state.get_next_state(action);
            if next_state.get_result() != 0 { score += 200.0; }
            scores.push(score);
        }

        // Return the softmax of the scores
        let max_score = scores.iter().fold(f64::MIN, |a, &b| a.max(b));
        let exp_scores: Vec<f64> = scores.iter().map(|s| ((s - max_score) as f64).exp()).collect();
        let sum_exp_scores: f64 = exp_scores.iter().sum();
        exp_scores.iter().map(|s| s / sum_exp_scores).collect()
    }

    fn get_next_state(&self, action: Self::Action) -> State {
        let mut next_board = self.board.clone();
        if let Some(row) = self.get_top_row(action) {
            next_board[row][action] = self.player;
        }
        let next_player = if self.player == 'X' { 'O' } else { 'X' };
        State {
            board: next_board,
            player: next_player,
        }
    }

    fn get_result(&self) -> i32 {
        // Check horizontal
        for row in &self.board {
            for col in 0..4 {
                if row[col] != ' ' && row[col] == row[col + 1] && row[col] == row[col + 2] && row[col] == row[col + 3] {
                    return -1;
                }
            }
        }

        // Check vertical
        for row in 0..3 {
            for col in 0..self.board[0].len() {
                if self.board[row][col] != ' ' && self.board[row][col] == self.board[row + 1][col] && self.board[row][col] == self.board[row + 2][col] && self.board[row][col] == self.board[row + 3][col] {
                    return -1;
                }
            }
        }

        // Check diagonal (down-right and down-left)
        for row in 0..3 {
            for col in 0..7 {
                if col <= 3 && self.board[row][col] != ' ' && self.board[row][col] == self.board[row + 1][col + 1] && self.board[row][col] == self.board[row + 2][col + 2] && self.board[row][col] == self.board[row + 3][col + 3] {
                    return -1;
                }
                if col >= 3 && self.board[row][col] != ' ' && self.board[row][col] == self.board[row + 1][col - 1] && self.board[row][col] == self.board[row + 2][col - 2] && self.board[row][col] == self.board[row + 3][col - 3] {
                    return -1;
                }
            }
        }

        0 // Draw or no winner yet
    }
}


pub fn play() -> Result<()> {
    // Terminal setup
    let mut stdout = stdout();
    terminal::enable_raw_mode()?;
    execute!(stdout, Hide, Clear(ClearType::All))?;

    // Game state initialization
    let mut root = Node::new(State::new(), None, None);
    let mut current_pos: <State as GameState>::Action = 0;

    // Main game loop
    loop {
        execute!(stdout, Clear(ClearType::All), MoveTo(0, 0))?;
        println!("{}\r", (0..7).map(|i| if i == current_pos { " [ ]" } else { "    " }).collect::<String>());
        println!("|---|---|---|---|---|---|---|\r");
        for row in root.borrow().state.board.iter().rev() {
            println!("| {} | {} | {} | {} | {} | {} | {} |\r", row[0], row[1], row[2], row[3], row[4], row[5], row[6]);
            println!("|---|---|---|---|---|---|---|\r");
        }
        println!("  1   2   3   4   5   6   7");

        if root.borrow().state.is_terminal() {
            break;
        }

        if root.borrow().state.player == 'X' {
            // Human turn
            if let Event::Key(key_event) = read()? {
                match key_event.code {
                    KeyCode::Left => if current_pos > 0 { current_pos -= 1; },
                    KeyCode::Right => if current_pos < 6 { current_pos += 1; },
                    KeyCode::Enter | KeyCode::Char(' ') => {
                        if root.borrow().state.get_legal_actions().contains(&current_pos) {
                            let next_state = root.borrow().state.get_next_state(current_pos);
                            root = Node::new(next_state, None, Some(current_pos));
                        }
                    },
                    KeyCode::Esc => break,
                    _ => {}
                }
            }
        } else {
            // AI turn
            let action = Node::best_action(&root, 1000);
            let next_state = root.borrow().state.get_next_state(action);
            root = Node::new(next_state, None, Some(action));
        }
    }

    let state = &root.borrow().state;
    let result_message = match state.get_result() {
        -1 => match state.player {
            'O' => "Player X wins!",
            'X' => "Player O wins!",
            _ => unreachable!(),
        },
        0 => "It's a draw!",
        _ => unreachable!(),
    };

    if state.is_terminal() {
        execute!(stdout, MoveTo(0, 0), Clear(ClearType::All))?;
        println!("{}", result_message);
        loop {
            if let Event::Key(_) = read()? { break; }
        }
    }

    terminal::disable_raw_mode()?;
    execute!(stdout, Show, MoveTo(0, 0), Clear(ClearType::All))?;
    Ok(())
}
