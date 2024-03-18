use std::fmt::Debug;
use std::rc::{Rc, Weak};
use std::cell::RefCell;
use std::hash::Hash;
use std::collections::HashMap;
use rand::distributions::WeightedIndex;
use rand::prelude::*;

pub trait GameState {
    type Action: Eq + Hash + Copy + Debug;

    fn is_terminal(&self) -> bool;
    fn get_player_turn(&self) -> i32;
    fn get_legal_actions(&self) -> Vec<Self::Action>;
    fn get_policy(&self, actions: &Vec<Self::Action>) -> Vec<f64>;
    fn get_next_state(&self, action: Self::Action) -> Self;
    fn get_result(&self) -> i32;
}

pub struct Node<State: GameState> {
    pub state: State,
    parent: Option<Weak<RefCell<Self>>>,
    parent_action: Option<State::Action>,
    children: Vec<Rc<RefCell<Self>>>,
    results: HashMap<i32, i32>,
    visit_count: i32,
    action_probs: HashMap<State::Action, f64>,
}

impl<State: GameState + Clone> Node<State> {
    pub fn new(state: State, parent: Option<Weak<RefCell<Self>>>, parent_action: Option<State::Action>) -> Rc<RefCell<Self>> {
        let actions = state.get_legal_actions();
        let weights = state.get_policy(&actions);
        let action_probs = actions.clone().into_iter().zip(weights.into_iter()).collect::<HashMap<_, _>>();
        Rc::new(RefCell::new(Node {
            state,
            parent,
            parent_action,
            children: Vec::new(),
            results: HashMap::new(),
            visit_count: 0,
            action_probs,
        }))
    }

    fn q(&self) -> f64 {
        if self.visit_count > 0 { (*self.results.get(&1).unwrap_or(&0) as f64 - *self.results.get(&-1).unwrap_or(&0) as f64) / (self.visit_count as f64) } else { -1. }
    }

    fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    // Step 1: Select and expand

    fn select_node(node: &Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        let mut current_node = Rc::clone(node);

        while !current_node.borrow().is_terminal() {
            if current_node.borrow().children.is_empty() {
                Node::expand(&current_node);
                return current_node.borrow().best_child();
            } else {
                let next_node = current_node.borrow().best_child();
                current_node = next_node;
            }
        }

        current_node
    }

    fn best_child(&self) -> Rc<RefCell<Self>> {
        self.children.iter().max_by(|a, b| {
            let a_score = a.borrow().uct_score(self.visit_count);
            let b_score = b.borrow().uct_score(self.visit_count);
            a_score.partial_cmp(&b_score).expect("Unable to compare scores, NaN or infinity encountered.")
        }).map(|node| Rc::clone(node)).expect("Unable to find best child node")
    }

    fn uct_score(&self, node_visit: i32) -> f64 {
        let action_prob = self.parent.as_ref().and_then(|parent_weak| parent_weak.upgrade()).map_or(0.0, |parent| {
            parent.borrow().action_probs.get(&self.parent_action.unwrap()).copied().unwrap_or(0.0)
        });

        let pb_c_init = 1.25;
        let pb_c_base = 19652.;
        let pb_c = pb_c_init + ((node_visit as f64 + pb_c_base + 1.) / pb_c_base).ln();
        let policy_score = (node_visit as f64).sqrt() * pb_c * action_prob / (self.visit_count as f64 + 1.);
        let value_score = (-self.q() + 1.) / 2.;
        value_score + policy_score
    }

    fn expand(node: &Rc<RefCell<Self>>) {
        let mut parent = node.borrow_mut();
        assert!(!parent.is_terminal(), "Attempted to expand a terminal node.");
        assert!(parent.children.is_empty(), "Attempted to re-expand a node.");

        let state = parent.state.clone();
        for &action in state.get_legal_actions().iter() {
            if let Some(&prob) = parent.action_probs.get(&action) {
                if prob > 1e-6 {
                    let child_state = state.get_next_state(action);
                    let child_node = Node::new(child_state, Some(Rc::downgrade(node)), Some(action));
                    parent.children.push(child_node);
                }
            }
        }
    }

    // Step 2: Rollout to the end of the game

    fn rollout(&self) -> i32 {
        let mut current_state = self.state.clone();
        while !current_state.is_terminal() {
            let actions = current_state.get_legal_actions();
            let weights = current_state.get_policy(&actions);
            let weight_sum: f64 = weights.iter().sum();
            assert!((weight_sum - 1.0).abs() < 1e-6, "Policy weights do not sum to 1: {:?}", weights);

            let dist = WeightedIndex::new(&weights).unwrap();
            let action = actions[dist.sample(&mut rand::thread_rng())];
            current_state = current_state.get_next_state(action);
        }
        let result = current_state.get_result();
        if self.state.get_player_turn() == current_state.get_player_turn() { result } else { -result }
    }

    // Step 3: Backpropagate the results

    fn backpropagate(node: &Rc<RefCell<Self>>, result: i32) {
        let mut current_node_option = Some(Rc::clone(node));
        let mut current_result = result;

        while let Some(current_node_rc) = current_node_option {
            let mut current_node = current_node_rc.borrow_mut();
            current_node.visit_count += 1;
            *current_node.results.entry(current_result).or_insert(0) += 1;

            current_result *= -1;  // Flip the result for the next level up, since it's from the opponent's perspective.
            current_node_option = match current_node.parent {
                Some(ref parent_weak) => parent_weak.upgrade(),
                None => None,
            };
        }
    }

    // Tree search

    pub fn best_action(root: &Rc<RefCell<Self>>, n_simulations: i32) -> State::Action {
        for _ in 0..n_simulations {
            let leaf_node = Node::select_node(root);
            let result = leaf_node.borrow().rollout();
            Node::backpropagate(&leaf_node, result);
        }

        // Select the action of the child with the highest visit count
        root.borrow().children.iter().max_by_key(|child| child.borrow().visit_count)
            .and_then(|child| child.borrow().parent_action).unwrap()
    }
}
