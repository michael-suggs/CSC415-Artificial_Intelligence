import artmap.*;
import java.io.BufferedReader;
import java.io.FileReader;

/**
 * Created by Michael Suggs on 27/11/16.
 */
public class artmapPark {
    /**
     * @param args
     */
    private static int correct = 0, incorrect = 0;

    public static void main(String[] args) throws Exception {
        Dartmap am = new Dartmap(22, 2);
        am.setNetworkType(Dartmap.DISTRIB);
        float in[] = new float[22];

        try(BufferedReader br = new BufferedReader(new FileReader("park_train.data"))) {
            String line = br.readLine();

            while (line != null) {
                String[] splitted = line.split("\\s+");
                for (int i = 0; i < splitted.length - 1; i++) {
                    in[i] = Float.parseFloat(splitted[i]);
                }
                am.train(in, Integer.parseInt(splitted[splitted.length-1]));
                line = br.readLine();
            }
        }

        try(BufferedReader br = new BufferedReader(new FileReader("park_test.data"))) {
            String line = br.readLine();

            while (line != null) {
                String[] splitted = line.split("\\s+");
                for (int i = 0; i < splitted.length - 1; i++) {
                    in[i] = Float.parseFloat(splitted[i]);
                }
                am.test(in);
                int result = am.getMaxOutputIndex();

                if (result == -1) {
                    System.out.print(" ");
                    incorrect++;
                } else {
                    System.out.print(result);
                    correct++;
                }

                line = br.readLine();
                System.out.println();
            }
        }

        System.out.printf("Correct: %d\nIncorrect: %d", correct, incorrect);

//        for (float y = 1.0f; y >= 0.0f; y -= 0.06f) {
//            for (float x = 0.0f; x <= 1.0f; x += 0.03f) {
//                in[0] = x; in[1] = y;
//                am.test (in);
//                int result = am.getMaxOutputIndex();
//                if (result == -1)
//                    System.out.print (" ");
//                else
//                    System.out.print (result);
//            }
//            System.out.println();
//        }
    }
}
