import artmap.*;

public class artmapTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Dartmap am = new Dartmap(2, 3);
		am.setNetworkType(Dartmap.FUZZY);
		float in[] = new float[2];

		in[0] = 0.1f;	in[1] = 0.1f;	am.train (in, 0);
		in[0] = 0.9f;	in[1] = 0.6f;	am.train (in, 1);
		in[0] = 0.4f;	in[1] = 0.9f;	am.train (in, 2);

		for (float y = 1.0f; y >= 0.0f; y -= 0.06f) {
			for (float x = 0.0f; x <= 1.0f; x += 0.03f) {
				in[0] = x; in[1] = y;
				am.test (in);
				int result = am.getMaxOutputIndex();
				if (result == -1)
					System.out.print (" ");
				else
					System.out.print (result);
			}
			System.out.println();
		}
	}

}
