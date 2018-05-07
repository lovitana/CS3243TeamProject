

import java.awt.Color;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PlayerSkeleton {

	/*
	 * Constants used as parameters for the AI
	 */
	
	//weights used by the Heuristic
	public static final float[] BEST_WEIGHTS = { -9.854448f, 2.4389153f,	-2.333226f,	0.36771116f,	-0.52315074f,
			-1.3534356f,	0.024638796f,	0.27249014f, 	-0.54726523f,	-1.4634881f, 	2.3102803f,  	-3.7318974f,
			-2.8147786f, 	-3.0926073f, 	-2.84025f, 	-3.6721375f, 	-2.2535586f, 	-2.2111578f, 	-2.9119258f,
			-3.4484754f, 	-3.2507586f, 	-7.856307f, 	-8.207972f, 	-0.047947817f, 	1.5860023f,	3.6743877f
	};

	/*
	 * Solvers: different AI with different parameters
	 * 
	 */
	
	// Informed Search :
	// solver that simply search for the best heuristic among the allowed moves
	public static final StartingSolver BASIC_SOLVER = new StartingSolver(new ImprovedHeuristics());
	
	//Adversial Search:
	// Solver using Min_max for a depth of 2
	public static final MinMaxSolver MINMAX_SOLVER = new MinMaxSolver(new ImprovedHeuristics(), 2);
	
	// Solver using Min_max for a depth of 3
	public static final MinMaxSolver DEEPER_MINMAX_SOLVER = new MinMaxSolver(new ImprovedHeuristics(), 3);


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
		public static int squaresAboveHoles(CopyState s) {
			int count = 0;
			for (int c = 0; c < CopyState.COLS; c++) {
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
		public static int groupedHoles(CopyState s) {
			int[][] grid = s.getField().clone();
			// fill up all non-holes

			// for each col, set all squares above to 1
			for (int c = 0; c < CopyState.COLS; c++) {
				int start = s.getTop()[c];
				for (int r = start; r < CopyState.ROWS; r++) {
					grid[r][c] = 1;
				}
			}

			// count number of hole groups (all connected holes make up 1 group)
			int numGroups = 0;
			for (int c = 0; c < CopyState.COLS; c++) {
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
		public static int sumOfHeights(CopyState s) {
			int sum = 0;
			for (int i = 0; i < CopyState.COLS; i++) {
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
		public static int maxHeightsDifference(CopyState s) {
			Integer[] box = new Integer[s.getTop().length];
			for (int i = 0; i < box.length; i++) {
				box[i] = s.getTop()[i];
			}
			List<Integer> tops = Arrays.asList(box);
			int highest = Collections.max(tops);
			int lowest = Collections.min(tops);
			return (highest - lowest);
		}

		public static int holes(CopyState s) {
			int holes = 0;
			for (int i = 0; i < CopyState.COLS; i++) {
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
		public float compute(CopyState next, float[] w);

		/**
		 * Gives an approximation of the current state based on the heuristic
		 * 
		 * @param state
		 *            The state to approximate
		 * @return The vector of feature values at the state
		 */
		public float[] featureValues(CopyState state); // FIXME ????

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

		private static final int LENGTH = CopyState.COLS + CopyState.COLS - 1 + 3;

		private static final int INDICE_COLS_WEIGHTS = 1;
		private static final int INDICE_COLS_DIFF_WEIGTHS = INDICE_COLS_WEIGHTS + CopyState.COLS;
		private static final int INDICE_MAX_HEIGHT_WEIGHT = INDICE_COLS_DIFF_WEIGTHS + CopyState.COLS - 1;
		private static final int INDICE_HOLES_WEIGHT = INDICE_MAX_HEIGHT_WEIGHT + 1;

		@Override
		public float compute(CopyState state, float[] weights) {

			if (state.lost) {
				return Float.NEGATIVE_INFINITY;
			}

			float heuristicValue = weights[0];
			int maxHeight = 0;

			for (int i = 0; i < CopyState.COLS; i++) {
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
			for (int i = 0; i < CopyState.COLS - 1; i++) {
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
		public float[] featureValues(CopyState state) {
			float[] values = new float[LENGTH];
			int maxHeight = 0;
			int holes = 0;
			for (int i = 0; i < CopyState.COLS; i++) {
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
			for (int i = 0; i < CopyState.COLS - 1; i++) {
				int diff = Math.abs(state.getTop()[i] - state.getTop()[i + 1]);
				values[CopyState.COLS + i] = diff;
			}
			return values;
		}

	}

	/**
	 * Uses both given heuristics & new heuristics FIXME change name?
	 */
	public static class ImprovedHeuristics extends GivenHeuristic {

		@Override
		public float compute(CopyState next, float[] w) {
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
		public float[] featureValues(CopyState state) {
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
		public float[] featureValues(CopyState s);

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
		public static CopyState copy(State s) {
			CopyState copy = new CopyState();
			copy.lost = s.lost;
			copy.nextPiece = s.nextPiece;
			int[][] field = s.getField();
			int[][] copyField = copy.getField();
			for (int i = 0; i < CopyState.ROWS; i++) {
				for (int j = 0; j < CopyState.COLS; j++) {
					copyField[i][j] = field[i][j];
				}
			}
			for (int i = 0; i < CopyState.COLS; i++) {
				copy.getTop()[i] = s.getTop()[i];
			}
			return copy;
		}
		public static CopyState nextState(State s, int[] move) {
			CopyState next = copy(s);

			next.makeMove(move);

			return next;
		}
		public static CopyState nextState(CopyState s, int[] move) {
			CopyState next = new CopyState();
			next.lost = s.lost;
			next.nextPiece = s.nextPiece;
			int[][] field = s.getField();
			int[][] copyField = next.getField();
			for (int i = 0; i < CopyState.ROWS; i++) {
				for (int j = 0; j < CopyState.COLS; j++) {
					copyField[i][j] = field[i][j];
				}
			}
			for (int i = 0; i < CopyState.COLS; i++) {
				next.getTop()[i] = s.getTop()[i];
			}

			next.makeMove(move);

			return next;
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
			CopyState copy = TetrisSolver.copy(s);
			List<Float> values = Arrays.stream(legalMoves).parallel().map(move -> getHeuristicValue(move, copy, w))
					.collect(Collectors.toList());

			statesEvaluated += values.size();

			int maxIdx = IntStream.range(0, values.size()).reduce(0, (i, j) -> values.get(i) > values.get(j) ? i : j);

			return maxIdx;
		}

		private float getHeuristicValue(int[] move, CopyState state, float[] weights) {
			CopyState next = TetrisSolver.nextState(state, move);
			return heuristic.compute(next, weights);
		}

		@Override
		public int weightsLength() {
			return heuristic.weightsLength();
		}

		public float[] featureValues(CopyState s) {
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
					CopyState next = TetrisSolver.nextState(s, move);
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
		private float minmax(CopyState s, int d, boolean maximizing, float[] weights, float alpha, float beta) {
			if (s.hasLost()) {
				return Float.NEGATIVE_INFINITY;
			}
			if (d <= 0) {
				return heuristic.compute(s, weights);
			}

			if (maximizing) {
				float best = Float.NEGATIVE_INFINITY;
				for (int[] move : s.legalMoves()) {
					CopyState next = TetrisSolver.nextState(s, move);
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
				for (int i = 0; i < CopyState.N_PIECES; i++) {
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

		public float[] featureValues(CopyState s) {
			return heuristic.featureValues(s);
		}
	}

	
	
	/*
	 * pick move function: not used
	 */
	public int pickMove(State s, int[][] legalMoves){
		return MINMAX_SOLVER.pickMove(s, legalMoves, BEST_WEIGHTS);
	}
	
	
	
	/**
	 * MAIN FUNCTION
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		State s = new State();
		new TFrame(s);
		TetrisSolver aI = MINMAX_SOLVER;
		long startTime = System.currentTimeMillis();
		int i = 0;
		while (!s.hasLost()) {
			s.makeMove(aI.pickMove(s, s.legalMoves(), BEST_WEIGHTS));
			 s.draw();
			 s.drawNext(0,0);
			 if(i%1_000 ==0) {
				 System.out.println("Played "+ i+" pieces in\n"
				 		+ ((System.currentTimeMillis()-startTime)/1000) + " Seconds");
			 }
			 i++;
			 
			//try { Thread.sleep(100); } catch (InterruptedException e) {
				// e.printStackTrace(); }
		}
		System.out.println("You have completed " + s.getRowsCleared() + " rows.");
	}






	/**
	 * Copy of State class.
	 * Permit to use a state without depending on the given State class
	 *
	 */
	public static class CopyState {
		public static final int COLS = 10;
		public static final int ROWS = 21;
		public static final int N_PIECES = 7;

		

		public boolean lost = false;
		
		
		

		
		public TLabel label;
		
		//current turn
		private int turn = 0;
		private int cleared = 0;
		
		//each square in the grid - int means empty - other values mean the turn it was placed
		private int[][] field = new int[ROWS][COLS];
		//top row+1 of each column
		//0 means empty
		private int[] top = new int[COLS];
		
		
		//number of next piece
		protected int nextPiece;
		
		
		
		//all legal moves - first index is piece type - then a list of 2-length arrays
		protected static int[][][] legalMoves = new int[N_PIECES][][];
		
		//indices for legalMoves
		public static final int ORIENT = 0;
		public static final int SLOT = 1;
		
		//possible orientations for a given piece type
		protected static int[] pOrients = {1,2,4,4,4,2,2};
		
		//the next several arrays define the piece vocabulary in detail
		//width of the pieces [piece ID][orientation]
		protected static int[][] pWidth = {
				{2},
				{1,4},
				{2,3,2,3},
				{2,3,2,3},
				{2,3,2,3},
				{3,2},
				{3,2}
		};
		//height of the pieces [piece ID][orientation]
		private static int[][] pHeight = {
				{2},
				{4,1},
				{3,2,3,2},
				{3,2,3,2},
				{3,2,3,2},
				{2,3},
				{2,3}
		};
		private static int[][][] pBottom = {
			{{0,0}},
			{{0},{0,0,0,0}},
			{{0,0},{0,1,1},{2,0},{0,0,0}},
			{{0,0},{0,0,0},{0,2},{1,1,0}},
			{{0,1},{1,0,1},{1,0},{0,0,0}},
			{{0,0,1},{1,0}},
			{{1,0,0},{0,1}}
		};
		private static int[][][] pTop = {
			{{2,2}},
			{{4},{1,1,1,1}},
			{{3,1},{2,2,2},{3,3},{1,1,2}},
			{{1,3},{2,1,1},{3,3},{2,2,2}},
			{{3,2},{2,2,2},{2,3},{1,2,1}},
			{{1,2,2},{3,2}},
			{{2,2,1},{2,3}}
		};
		
		//initialize legalMoves
		{
			//for each piece type
			for(int i = 0; i < N_PIECES; i++) {
				//figure number of legal moves
				int n = 0;
				for(int j = 0; j < pOrients[i]; j++) {
					//number of locations in this orientation
					n += COLS+1-pWidth[i][j];
				}
				//allocate space
				legalMoves[i] = new int[n][2];
				//for each orientation
				n = 0;
				for(int j = 0; j < pOrients[i]; j++) {
					//for each slot
					for(int k = 0; k < COLS+1-pWidth[i][j];k++) {
						legalMoves[i][n][ORIENT] = j;
						legalMoves[i][n][SLOT] = k;
						n++;
					}
				}
			}
		
		}
		
		
		public int[][] getField() {
			return field;
		}

		public int[] getTop() {
			return top;
		}

	    public int[] getpOrients() {
	        return pOrients;
	    }
	    
	    public  int[][] getpWidth() {
	        return pWidth;
	    }

	    public  int[][] getpHeight() {
	        return pHeight;
	    }

	    public  int[][][] getpBottom() {
	        return pBottom;
	    }

	    public  int[][][] getpTop() {
	        return pTop;
	    }


		public int getNextPiece() {
			return nextPiece;
		}
		
		public boolean hasLost() {
			return lost;
		}
		
		public int getRowsCleared() {
			return cleared;
		}
		
		public int getTurnNumber() {
			return turn;
		}
		
		
		
		//constructor
		public CopyState() {
			nextPiece = randomPiece();

		}
		
		//random integer, returns 0-6
		private int randomPiece() {
			return (int)(Math.random()*N_PIECES);
		}
		


		
		//gives legal moves for 
		public int[][] legalMoves() {
			return legalMoves[nextPiece];
		}
		
		//make a move based on the move index - its order in the legalMoves list
		public void makeMove(int move) {
			makeMove(legalMoves[nextPiece][move]);
		}
		
		//make a move based on an array of orient and slot
		public void makeMove(int[] move) {
			makeMove(move[ORIENT],move[SLOT]);
		}
		
		//returns false if you lose - true otherwise
		public boolean makeMove(int orient, int slot) {
			turn++;
			//height if the first column makes contact
			int height = top[slot]-pBottom[nextPiece][orient][0];
			//for each column beyond the first in the piece
			for(int c = 1; c < pWidth[nextPiece][orient];c++) {
				height = Math.max(height,top[slot+c]-pBottom[nextPiece][orient][c]);
			}
			
			//check if game ended
			if(height+pHeight[nextPiece][orient] >= ROWS) {
				lost = true;
				return false;
			}

			
			//for each column in the piece - fill in the appropriate blocks
			for(int i = 0; i < pWidth[nextPiece][orient]; i++) {
				
				//from bottom to top of brick
				for(int h = height+pBottom[nextPiece][orient][i]; h < height+pTop[nextPiece][orient][i]; h++) {
					field[h][i+slot] = turn;
				}
			}
			
			//adjust top
			for(int c = 0; c < pWidth[nextPiece][orient]; c++) {
				top[slot+c]=height+pTop[nextPiece][orient][c];
			}
			
			int rowsCleared = 0;
			
			//check for full rows - starting at the top
			for(int r = height+pHeight[nextPiece][orient]-1; r >= height; r--) {
				//check all columns in the row
				boolean full = true;
				for(int c = 0; c < COLS; c++) {
					if(field[r][c] == 0) {
						full = false;
						break;
					}
				}
				//if the row was full - remove it and slide above stuff down
				if(full) {
					rowsCleared++;
					cleared++;
					//for each column
					for(int c = 0; c < COLS; c++) {

						//slide down all bricks
						for(int i = r; i < top[c]; i++) {
							field[i][c] = field[i+1][c];
						}
						//lower the top
						top[c]--;
						while(top[c]>=1 && field[top[c]-1][c]==0)	top[c]--;
					}
				}
			}
		

			//pick a new piece
			nextPiece = randomPiece();
			

			
			return true;
		}
		
		public void draw() {
			label.clear();
			label.setPenRadius();
			//outline board
			label.line(0, 0, 0, ROWS+5);
			label.line(COLS, 0, COLS, ROWS+5);
			label.line(0, 0, COLS, 0);
			label.line(0, ROWS-1, COLS, ROWS-1);
			
			//show bricks
					
			for(int c = 0; c < COLS; c++) {
				for(int r = 0; r < top[c]; r++) {
					if(field[r][c] != 0) {
						drawBrick(c,r);
					}
				}
			}
			
			for(int i = 0; i < COLS; i++) {
				label.setPenColor(Color.red);
				label.line(i, top[i], i+1, top[i]);
				label.setPenColor();
			}
			
			label.show();
			
			
		}
		
		public static final Color brickCol = Color.gray; 
		
		private void drawBrick(int c, int r) {
			label.filledRectangleLL(c, r, 1, 1, brickCol);
			label.rectangleLL(c, r, 1, 1);
		}
		
		public void drawNext(int slot, int orient) {
			for(int i = 0; i < pWidth[nextPiece][orient]; i++) {
				for(int j = pBottom[nextPiece][orient][i]; j <pTop[nextPiece][orient][i]; j++) {
					drawBrick(i+slot, j+ROWS+1);
				}
			}
			label.show();
		}
		
		//visualization
		//clears the area where the next piece is shown (top)
		public void clearNext() {
			label.filledRectangleLL(0, ROWS+.9, COLS, 4.2, TLabel.DEFAULT_CLEAR_COLOR);
			label.line(0, 0, 0, ROWS+5);
			label.line(COLS, 0, COLS, ROWS+5);
		}
		

		

	}






}



