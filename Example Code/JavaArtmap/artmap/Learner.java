package artmap;

public interface Learner {
        public void    train (float[] in, int cl);
        public float[] test  (float[] in);
		public void    clear ();
}

