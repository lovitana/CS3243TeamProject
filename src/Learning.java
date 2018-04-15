package src;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.Writer;

public class Learning {

	public static void main(String[] args) {

		try(Writer writer = new BufferedWriter(new FileWriter("count.csv", false))){}
		catch(Exception e){
			throw new Error();
		}
		
		float[] best = { 1.7035127f, 2.0450842f, -0.6609158f, 0.46027747f, -2.5351565f, 0.9790008f, 0.64786285f,
				-0.2330531f, 0.40242445f, -0.54276633f, 1.5980941f, -3.8651803f, -2.020836f, -3.5747838f, -4.352761f,
				-3.1158588f, -2.380412f, -3.3974721f, -1.8160414f, -3.9547849f, -1.3370675f, -10.818589f, -6.772875f,
				-0.35120985f, -0.037590336f, 0.4731508f };
		TetrisLearner learner = new ArrangedSALearner(3014);
		float[] optimalW = learner.learn(PlayerSkeleton.IMPROVED_BASIC_SOLVER, 10000, 400_000, 8,best);
		System.out.println("END :");
		for (float w : optimalW) {
			System.out.println(w + "f , ");
		}
	}

}
