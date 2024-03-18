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
            board: vec![vec![' '; 3]; 3],
            player: 'X',
        }
    }
}

impl GameState for State {
    type Action = (usize, usize);

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
        let mut actions = Vec::new();

        for (i, row) in self.board.iter().enumerate() {
            for (j, &cell) in row.iter().enumerate() {
                if cell == ' ' {
                    actions.push((i, j));
                }
            }
        }

        actions
    }

    fn get_policy(&self, actions: &Vec<Self::Action>) -> Vec<f64> {
        vec![1.0 / actions.len() as f64; actions.len()]
    }

    fn get_next_state(&self, action: Self::Action) -> State {
        let mut next_board = self.board.clone();
        next_board[action.0][action.1] = self.player;
        let next_player = if self.player == 'X' { 'O' } else { 'X' };

        State {
            board: next_board,
            player: next_player,
        }
    }

    fn get_result(&self) -> i32 {
        for i in 0..3 {
            if self.board[i][0] == self.board[i][1] && self.board[i][1] == self.board[i][2] && self.board[i][0] != ' ' {
                return -1;
            }
            if self.board[0][i] == self.board[1][i] && self.board[1][i] == self.board[2][i] && self.board[0][i] != ' ' {
                return -1;
            }
        }
        if (self.board[0][0] == self.board[1][1] && self.board[1][1] == self.board[2][2] && self.board[0][0] != ' ')
            || (self.board[0][2] == self.board[1][1] && self.board[1][1] == self.board[2][0] && self.board[0][2] != ' ') {
            return -1;
        }

        0
    }
}


pub fn play() -> Result<()> {
    // Terminal setup
    let mut stdout = stdout();
    terminal::enable_raw_mode()?;
    execute!(stdout, Hide, Clear(ClearType::All))?;

    // Game state initialization
    let mut root = Node::new(State::new(), None, None);
    let mut current_pos: <State as GameState>::Action = (0, 0);

    // Main game loop
    loop {
        execute!(stdout, Clear(ClearType::All), MoveTo(0, 0))?;
        println!("+---+---+---+\r");
        for i in 0..3 {
            for j in 0..3 {
                let cell = root.borrow().state.board[i][j];
                let marker = if (i, j) == current_pos { format!("[{}]", cell) } else { format!(" {} ", cell) };
                print!("|{}", marker);
            }
            println!("|\r");
            println!("+---+---+---+\r")
        }

        if root.borrow().state.is_terminal() {
            break;
        }

        if root.borrow().state.player == 'X' {
            // Human turn
            if let Event::Key(key_event) = read()? {
                match key_event.code {
                    KeyCode::Up => if current_pos.0 > 0 { current_pos.0 -= 1; },
                    KeyCode::Down => if current_pos.0 < 2 { current_pos.0 += 1; },
                    KeyCode::Left => if current_pos.1 > 0 { current_pos.1 -= 1; },
                    KeyCode::Right => if current_pos.1 < 2 { current_pos.1 += 1; },
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
