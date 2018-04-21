package src;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PlayerSkeleton {

	/*
	 * Constants used as parameters for the AI
	 */
	public static final float[] DUMBS_WEIGHTS = new float[GivenHeuristic.LENGTH];
	public static final float[] DUMB_EXPERIMENTAL_WEIGHTS = new float[new ImprovedHeuristics().weightsLength()];

	// Manually entered weights
	public static final float[] BASICS_WEIGHTS = { 0, // weight 0
			-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // columns weights
			-1, -1, -1, -1, -1, -1, -1, -1, -1, // differences between columns
												// weights
			-1, //
			-50 };

	// FIXME This constant is a draft of previous result, delete this before
	// submitting the code
	public static final float[] COMPUTED_WEIGHTS = { -4.0f, 0.0f, -1.0f, 0.0f, -2.0f, -1.0f, 0.0f, 0.0f, -1.0f, -2.0f,
			1.0f, -3.0f, -1.0f, -2.0f, -2.0f, -2.0f, -1.0f, -2.0f, -2.0f, -1.0f, -3.0f, -9.0f };

	public static final float[] BEST_WEIGHTS = { -2.064638f, 0.99241304f, -0.9892719f, -0.022724867f, -0.9527318f,
			-0.021005929f, 0.012356937f, -0.016091943f, -0.0025559664f, -1.0049679f, 0.9978579f, -2.940033f, -1.950203f,
			-2.0007744f, -1.956233f, -1.9986322f, -1.9524348f, -1.9964991f, -1.97714f, -2.9492042f, -0.009090185f,
			-9.940321f }; // loic 80'000

	public static final float[] BEST_WEIGHTS_IMPROVED = { 1.7035127f, 2.0450842f, -0.6609158f, 0.2362841f, -2.5351565f,
			0.9790008f, 0.64786285f, -0.42757857f, 0.62548554f, -0.54276633f, 1.5980941f, -3.8651803f, -2.020836f,
			-3.5747838f, -4.352761f, -3.326226f, -2.380412f, -3.5919976f, -1.8160414f, -4.177846f, -1.548367f,
			-10.818589f, -6.1607633f, -0.14084265f, 0.17277686f, 0.4731508f

	};
	
	public static final float[] BEST_WEIGHTS_FINAL = { -9.854448f, 2.4389153f,	-2.333226f,	0.36771116f,	-0.52315074f,
			-1.3534356f,	0.024638796f,	0.27249014f, 	-0.54726523f,	-1.4634881f, 	2.3102803f,  	-3.7318974f,
			-2.8147786f, 	-3.0926073f, 	-2.84025f, 	-3.6721375f, 	-2.2535586f, 	-2.2111578f, 	-2.9119258f,
			-3.4484754f, 	-3.2507586f, 	-7.856307f, 	-8.207972f, 	-0.047947817f, 	1.5860023f,	3.6743877f
	};

	/*
	 * Solvers: different AI with different parameters
	 * 
	 */
	public static final RandomSolver RANDOM_SOLVER = new RandomSolver();// FIXME
	public static final StartingSolver BASIC_SOLVER = new StartingSolver(new GivenHeuristic());
	public static final MinMaxSolver MINMAX_SOLVER = new MinMaxSolver(new GivenHeuristic(), 2);
	public static final MinMaxSolver DEEPER_MINMAX_SOLVER = new MinMaxSolver(new GivenHeuristic(), 3);
	public static final TetrisSolver IMPROVED_BASIC_SOLVER = new StartingSolver(new ImprovedHeuristics()); // FIXME
	public static final MinMaxSolver IMPROVED_MINMAX_SOLVER = new MinMaxSolver(new ImprovedHeuristics(), 2);
	public static final MinMaxSolver IMPROVED_DEEPER_MINMAX_SOLVER = new MinMaxSolver(new ImprovedHeuristics(), 3);
	// change
	// name!!!

	/**
	 * Tool interface containing useful functions used by the heuristics
	 *
	 */
	public static interface HeuristicTool {

		/**
		 * compute the number of filled tiles above every holes
		 * 
		 * @param s
		 *            the state of the game
		 * @return number of filled tiles above every holes
		 */
		public static int squaresAboveHoles(State s) {
			int count = 0;
			for (int c = 0; c < State.COLS; c++) {
				int end = s.getTop()[c];
				for (int r = 0; r < end; r++) {
					boolean isHole = s.getField()[r][c] == 0;
					if (isHole) {
						count += end - r;
						break;
					}
				}
			}
			return count;
		}

		/**
		 * count the number of grouped holes in the state
		 * 
		 * @param s
		 *            the state of the game
		 * @return number of grouped holes.
		 */
		public static int groupedHoles(State s) {
			int[][] grid = s.getField().clone();
			// fill up all non-holes

			// for each col, set all squares above to 1
			for (int c = 0; c < State.COLS; c++) {
				int start = s.getTop()[c];
				for (int r = start; r < State.ROWS; r++) {
					grid[r][c] = 1;
				}
			}

			// count number of hole groups (all connected holes make up 1 group)
			int numGroups = 0;
			for (int c = 0; c < State.COLS; c++) {
				for (int r = 0; r < s.getTop()[c]; r++) {
					boolean isNotHole = grid[r][c] > 0;
					if (isNotHole)
						continue;
					numGroups++;
					fillNeighbors(grid, r, c);
				}
			}

			return numGroups;
		}

		/**
		 * help function, that fill the neighbors of a given coordinate
		 * 
		 * @param grid
		 * @param y
		 *            coordinate y
		 * @param x
		 *            coordinate x
		 */
		public static void fillNeighbors(int[][] grid, int y, int x) {
			if (grid[y][x] == 0) {
				grid[y][x] = 1;

				// explore up, down, left, right recursively
				int left = x - 1;
				int right = x + 1;
				int down = y - 1;
				int up = y + 1;
				if (left >= 0)
					fillNeighbors(grid, y, left);
				if (right < grid[0].length)
					fillNeighbors(grid, y, right);
				if (down >= 0)
					fillNeighbors(grid, down, x);
				if (up < grid.length)
					fillNeighbors(grid, up, x);
			}
		}

		/**
		 * Compute the sum of heights
		 * 
		 * @param s
		 *            the state of the game
		 * @return the sum
		 */
		public static int sumOfHeights(State s) {
			int sum = 0;
			for (int i = 0; i < State.COLS; i++) {
				sum += s.getTop()[i];
			}
			return sum;
		}

		/**
		 * compute the difference between the highest column and the smallest
		 * 
		 * @param s
		 *            the state of the game
		 * @return
		 */
		public static int maxHeightsDifference(State s) {
			Integer[] box = new Integer[s.getTop().length];
			for (int i = 0; i < box.length; i++) {
				box[i] = s.getTop()[i];
			}
			List<Integer> tops = Arrays.asList(box);
			int highest = Collections.max(tops);
			int lowest = Collections.min(tops);
			return (highest - lowest);
		}

		public static int holes(State s) {
			int holes = 0;
			for (int i = 0; i < State.COLS; i++) {
				int height = s.getTop()[i];
				for (int j = 0; j < height - 1; j++) {
					if (s.getField()[j][i] == 0) {
						holes++;
					}
				}
			}
			return holes;
		}

	}

	/**
	 * Interface of heuristic
	 *
	 */
	public static interface Heuristic {

		/**
		 * Compute the heuristic value of a state
		 * 
		 * @param next
		 *            The state to with the heuristic to compute
		 * @param w
		 *            the weights parameters of the heuristic
		 * @return the heuristic result
		 */
		public float compute(State next, float[] w);

		/**
		 * Gives an approximation of the current state based on the heuristic
		 * 
		 * @param state
		 *            The state to approximate
		 * @return The vector of feature values at the state
		 */
		public float[] featureValues(State state); // FIXME ????

		/**
		 * @return the number of weight used by the solver
		 */
		public int weightsLength();
	}

	/**
	 * class representing the heuristic given ins the projects instruction 0 ->
	 * bias next num_cols -> height of walls next num_cols -> difference between
	 * adj columns next 1 -> max col height next 1 -> num holes in wall
	 */
	public static class GivenHeuristic implements Heuristic {

		private static final int LENGTH = State.COLS + State.COLS - 1 + 3;

		private static final int INDICE_COLS_WEIGHTS = 1;
		private static final int INDICE_COLS_DIFF_WEIGTHS = INDICE_COLS_WEIGHTS + State.COLS;
		private static final int INDICE_MAX_HEIGHT_WEIGHT = INDICE_COLS_DIFF_WEIGTHS + State.COLS - 1;
		private static final int INDICE_HOLES_WEIGHT = INDICE_MAX_HEIGHT_WEIGHT + 1;

		@Override
		public float compute(State state, float[] weights) {

			if (state.lost) {
				return Float.NEGATIVE_INFINITY;
			}

			float heuristicValue = weights[0];
			int maxHeight = 0;

			for (int i = 0; i < State.COLS; i++) {
				// height
				int height = state.getTop()[i];
				heuristicValue += weights[INDICE_COLS_WEIGHTS + i] * height;
				if (height > maxHeight) {
					maxHeight = height;
				}
			}
			heuristicValue += maxHeight * weights[INDICE_MAX_HEIGHT_WEIGHT];

			// holes
			heuristicValue += HeuristicTool.holes(state) * weights[INDICE_HOLES_WEIGHT];

			// differences
			for (int i = 0; i < State.COLS - 1; i++) {
				int diff = Math.abs(state.getTop()[i] - state.getTop()[i + 1]);

				heuristicValue += weights[INDICE_COLS_DIFF_WEIGTHS + i] * diff;
			}

			return heuristicValue;
		}

		@Override
		public int weightsLength() {
			return LENGTH;
		}

		@Override
		public float[] featureValues(State state) {
			float[] values = new float[LENGTH];
			int maxHeight = 0;
			int holes = 0;
			for (int i = 0; i < State.COLS; i++) {
				// height
				int height = state.getTop()[i];
				values[i] = height;
				if (height > maxHeight) {
					maxHeight = height;
				}

				// holes
				for (int j = 0; j < height - 1; j++) {
					if (state.getField()[j][i] == 0) {
						holes++;
					}
				}
			}
			values[LENGTH - 1] = holes;
			values[LENGTH - 2] = maxHeight;

			// differences
			for (int i = 0; i < State.COLS - 1; i++) {
				int diff = Math.abs(state.getTop()[i] - state.getTop()[i + 1]);
				values[State.COLS + i] = diff;
			}
			return values;
		}

	}

	/**
	 * Uses both given heuristics & new heuristics FIXME change name?
	 */
	public static class ImprovedHeuristics extends GivenHeuristic {

		@Override
		public float compute(State next, float[] w) {
			float oldScore = super.compute(next, w);

			int nextIndex = super.weightsLength();
			float newScore = 0;

			// grouped holes
			newScore += HeuristicTool.groupedHoles(next) * w[nextIndex];
			nextIndex++;

			// sum of heights
			newScore += HeuristicTool.sumOfHeights(next) * w[nextIndex];
			nextIndex++;

			// max difference
			newScore += HeuristicTool.maxHeightsDifference(next) * w[nextIndex];
			nextIndex++;

			// squares above holes
			newScore += HeuristicTool.squaresAboveHoles(next) * w[nextIndex];
			nextIndex++;

			if (next.hasLost())
				newScore = Float.NEGATIVE_INFINITY;

			return oldScore + newScore;
		}

		@Override
		public int weightsLength() {
			return super.weightsLength() + 4;
		}

		@Override
		public float[] featureValues(State state) {
			float[] oldValues = super.featureValues(state);

			float[] newValues = new float[oldValues.length + weightsLength()];
			System.arraycopy(oldValues, 0, newValues, 0, oldValues.length);

			int nextIndex = oldValues.length;
			newValues[nextIndex] = HeuristicTool.groupedHoles(state);
			nextIndex++;

			newValues[nextIndex] = HeuristicTool.sumOfHeights(state);
			nextIndex++;

			newValues[nextIndex] = HeuristicTool.maxHeightsDifference(state);
			nextIndex++;

			newValues[nextIndex] = HeuristicTool.squaresAboveHoles(state);
			nextIndex++;

			return newValues;
		}
	}

	/**
	 * Interface that represent an AI for Tetris
	 */
	public static interface TetrisSolver {

		/**
		 * Return the best move given the current state of the Tetris board
		 * 
		 * @param s
		 *            the current state of the game
		 * @param legalMoves
		 *            authorized moves that can be played on this state
		 * @return the index of the selected move from legalMoves
		 */
		public int pickMove(State s, int[][] legalMoves, float[] weights);

		/**
		 * Return an approximation of the current state as a vector of the
		 * features of the heuristic used by this solver, if any.
		 * 
		 * @param s
		 *            The state to approximate
		 * @return The feature vector of the state
		 */
		public float[] featureValues(State s);

		/**
		 * @return the number of weight used by the solver
		 */
		public int weightsLength();

		/**
		 * Create a new State of the game after the selected move This does not
		 * affect the given state, thus we can call this function on the current
		 * state without changing it.
		 * 
		 * @param s
		 *            Current state of the game
		 * @param move
		 *            move played
		 * @return the state of the game after playing the selected move
		 */
		public static State nextState(State s, int[] move) {
			State next = new State();
			next.lost = s.lost;
			next.nextPiece = s.nextPiece;
			int[][] field = s.getField();
			int[][] copyField = next.getField();
			for (int i = 0; i < State.ROWS; i++) {
				for (int j = 0; j < State.COLS; j++) {
					copyField[i][j] = field[i][j];
				}
			}
			for (int i = 0; i < State.COLS; i++) {
				next.getTop()[i] = s.getTop()[i];
			}

			next.makeMove(move);

			return next;
		}
	}

	/**
	 * Solver that return a random move
	 *
	 */
	public static final class RandomSolver implements TetrisSolver {

		@Override
		public int pickMove(State s, int[][] legalMoves, float[] w) {
			return (int) (Math.random() * legalMoves.length);
		}

		@Override
		public int weightsLength() {
			return 0;
		}

		public float[] featureValues(State s) {
			return null;
		}

	}

	/**
	 * Solver using the heuristic function at depth 1
	 *
	 */
	public static final class StartingSolver implements TetrisSolver {

		// for benchmarking
		public int statesEvaluated = 0;

		private final Heuristic heuristic;

		/**
		 * Default constructor, initialize all weights to 0
		 */
		public StartingSolver(Heuristic heuristic) {
			this.heuristic = heuristic;
		}

		@Override
		public int pickMove(State s, int[][] legalMoves, float[] w) {
			if (w.length != weightsLength()) {
				throw new IllegalArgumentException(
						"wrong number of weights: " + w.length + ". Expected: " + weightsLength());
			}
			List<Float> values = Arrays.stream(legalMoves).parallel().map(move -> getHeuristicValue(move, s, w))
					.collect(Collectors.toList());

			statesEvaluated += values.size();

			int maxIdx = IntStream.range(0, values.size()).reduce(0, (i, j) -> values.get(i) > values.get(j) ? i : j);

			return maxIdx;
		}

		private float getHeuristicValue(int[] move, State state, float[] weights) {
			State next = TetrisSolver.nextState(state, move);
			return heuristic.compute(next, weights);
		}

		@Override
		public int weightsLength() {
			return heuristic.weightsLength();
		}

		public float[] featureValues(State s) {
			return heuristic.featureValues(s);
		}

	}

	/**
	 * Solver using MinMax Algorithm
	 *
	 */
	public static final class MinMaxSolver implements TetrisSolver {

		private final Heuristic heuristic;
		private final int depth;

		/**
		 * @param heur
		 * @param depth
		 *            depth used for the minmax algorithm,
		 */
		public MinMaxSolver(Heuristic heur, int depth) {
			heuristic = heur;
			this.depth = depth;
		}

		@Override
		public int pickMove(State s, int[][] legalMoves, float[] weights) {
			if (weights.length != weightsLength()) {
				throw new IllegalArgumentException(
						"wrong number of weights: " + weights.length + ". Expected: " + weightsLength());
			}
			float max = Float.NEGATIVE_INFINITY;
			int bestMove = 0;

			int d = depth - 1;
			while (max == Float.NEGATIVE_INFINITY && d >= 0) {
				int n = 0;
				for (int[] move : legalMoves) {
					State next = TetrisSolver.nextState(s, move);
					float heuristicValue = minmax(next, d, false, weights, max, Float.POSITIVE_INFINITY);

					if (heuristicValue > max) {
						max = heuristicValue;
						bestMove = n;
					}
					n++;
				}
				d -= 1;
			}
			return bestMove;
		}

		@Override
		public int weightsLength() {
			return heuristic.weightsLength();
		}

		/**
		 * Minmax algorithm applied for tetris. Max tries to play the best legal
		 * move Min tries to select the most annoying piece as the next piece
		 * playable
		 * 
		 * @param s
		 *            state of the node
		 * @param d
		 *            depth
		 * @param maximizing
		 *            boolean true if we try to maximize
		 * @param weights
		 *            weights
		 * @return the MINMAX best heuristic value
		 */
		private float minmax(State s, int d, boolean maximizing, float[] weights, float alpha, float beta) {
			if (s.hasLost()) {
				return Float.NEGATIVE_INFINITY;
			}
			if (d <= 0) {
				return heuristic.compute(s, weights);
			}

			if (maximizing) {
				float best = Float.NEGATIVE_INFINITY;
				for (int[] move : s.legalMoves()) {
					State next = TetrisSolver.nextState(s, move);
					float v = minmax(next, d - 1, false, weights, alpha, beta);
					if (v > best) {
						best = v;
					}
					if (alpha < v) {
						alpha = v;
					}
					if (beta <= alpha) {
						break;
					}
				}
				return best;
			} else {
				float best = Float.POSITIVE_INFINITY;
				for (int i = 0; i < State.N_PIECES; i++) {
					s.nextPiece = i;
					float v = minmax(s, d, true, weights, alpha, beta);
					if (v < best) {
						best = v;
					}
					if (v < beta) {
						beta = v;
					}
					if (beta <= alpha) {
						break;
					}
				}
				return best;
			}
		}

		public float[] featureValues(State s) {
			return heuristic.featureValues(s);
		}
	}

	// FIXME we need to clean this before submitting
	/**
	 * MAIN FUNCTION
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		State s = new State();
		// new TFrame(s);
		TetrisSolver aI = MINMAX_SOLVER;
		long start = System.currentTimeMillis();
		while (!s.hasLost()) {
			s.makeMove(aI.pickMove(s, s.legalMoves(), BEST_WEIGHTS));
			// s.draw();
			// s.drawNext(0,0);
			/*
			 * try { Thread.sleep(100); } catch (InterruptedException e) {
			 * e.printStackTrace(); }
			 */
			if (s.getRowsCleared() % 1000 == 0) {
				System.out.println(" lines cleared " + s.getRowsCleared());
			}
		}
		long end = System.currentTimeMillis();
		long diff = end - start;
		System.out.println("it takes " + diff + " milliseconds");
		System.out.println("You have completed " + s.getRowsCleared() + " rows.");
	}

}
