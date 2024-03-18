mod mcts;
mod ttt;
mod connect4;
use std::env;
use crossterm::Result;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("ttt") => ttt::play(),
        Some("connect4") => connect4::play(),
        _ => Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid game selection").into()),
    }
}
