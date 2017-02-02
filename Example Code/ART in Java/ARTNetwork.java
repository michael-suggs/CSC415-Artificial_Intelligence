package kcdc.nn.art;

import kcdc.nn.*;

public interface ARTNetwork extends NeuralNetwork
{
  public void epoch();
  public boolean train(PatternSet pattern_set, int max_epochs);
  public boolean train(Pattern pattern);
  public boolean test(PatternSet pattern_set);
  public boolean test(Pattern pattern);
  public void activate(Pattern pattern);
  public int getBestMatch();
  public int getWinnerIndex();
  public double getConfidence();

  public void setVigilance(double v);
  public double getVigilance();

  public void setLearningRate(double r);
  public double getLearningRate();

  public double getOutput(int i);
  public Pattern getOutputPattern();
  public Pattern getInputPattern();

  public int getNumResets();
  public int getNumResizes();

  public void addListener(ARTListener al);
  public void removeListener(ARTListener al);
}
