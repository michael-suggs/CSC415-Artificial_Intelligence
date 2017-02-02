package kcdc.nn.art;

import kcdc.nn.*;

public class CopyART implements ARTNetwork
{
  Pattern input_pattern = null;
  Pattern output_pattern = null;

  int size = 0;

  boolean initialized = true;
  boolean done = false;

  boolean activated = false;

  double vigilance = 0.0;

  int num_epochs = 0;

  int test_num_correct = 0;
  int test_num_unanswered = 0;
  int test_num_errors = 0;

  public CopyART() {}

  public CopyART(int num_inputs, int input_style)
  {
  }

  public void epoch() { num_epochs++; }

  public boolean train(PatternSet pattern_set, int max_epochs)
  {
    boolean trained = true;

    for (int i = 0; i < pattern_set.getSize(); i++)
      trained = trained && train(pattern_set.getPattern(i));

    return trained;
  }

  public boolean train(Pattern pattern)
  {
    activate(pattern);

    if (pattern.getSize() > size)
    {
      size = pattern.getSize();
      multicaster.artResize(new ARTEvent(this, ARTEvent.ART_RESIZE));
    }

    return true;
  }

  public boolean test(PatternSet pattern_set)
  {
    boolean tested = true;

    for (int i = 0; i < pattern_set.getSize(); i++)
      tested = tested && test(pattern_set.getPattern(i));

    return tested;
  }

  public boolean test(Pattern pattern)
  {
    activate(pattern);
    return true;
  }

  public void activate(Pattern pattern)
  {
    input_pattern = pattern;
    output_pattern = pattern;
    activated = true;
  }

  public int getBestMatch()
  {
    if (!activated) return -1;

    int best_index = -1;
    double best_output = -1.0;

    for (int i = 0; i < output_pattern.getSize(); i++)
    {
      if (output_pattern.getInput(i) > best_output)
      {
	best_index = i;
	best_output = output_pattern.getInput(i);
      }
    }

    return best_index;
  }

  public int getWinnerIndex()
  { return getBestMatch(); }

  public double getConfidence()
  { return 1.0; }

  public void setVigilance(double v)
  { vigilance = v; }

  public double getVigilance()
  { return vigilance; }

  public void setLearningRate(double r)
  {}

  public double getLearningRate()
  { return 0.0; }

  public double getOutput(int i)
  { return output_pattern.getInput(i); }

  public Pattern getOutputPattern()
  { return output_pattern; }

  public void setInputPattern(Pattern pattern)
  { input_pattern = pattern; }

  public Pattern getInputPattern()
  { return input_pattern; }

  public int getNumResets()
  { return 0; }

  public int getNumResizes()
  { return 0; }

  //// NeuralNetwork Interface ////

  public void init()
  { initialized = true; }

  public int getSize()
  { return size; }

  public int getNumEpochs()
  { return num_epochs; }

  public boolean isInit()
  { return initialized; }

  public boolean isDone()
  { return done; }

  public void setNumInputs(int n)
  {}

  public int getNumInputs()
  { if (input_pattern != null) return input_pattern.getSize(); else return 0; }

  int input_style = 0;

  public void setInputStyle(int s)
  { input_style = s; }

  public int getInputStyle()
  { return input_style; }

  int input_type = 0;

  public void setInputType(int t)
  { input_type = t; }

  public int getInputType()
  { return input_type; }

  public void setNumOutputs(int n)
  {}

  public int getNumOutputs()
  { if (output_pattern != null) return output_pattern.getSize(); else return 0; }

  int output_style = 0;

  public void setOutputStyle(int s)
  { output_style = s; }

  public int getOutputStyle()
  { return output_style; }

  int output_type = 0;

  public void setOutputType(int t)
  { output_type = t; }

  public int getOutputType()
  { return output_type; }

  public int getNumCorrect()
  { return test_num_correct; }

  public int getNumErrors()
  { return test_num_errors; }

  public int getNumUnanswered()
  { return test_num_unanswered; }

  //// Event processing ////

  ARTEventMulticaster multicaster = new ARTEventMulticaster();

  public void addListener(ARTListener al)
  { multicaster.addListener(al); }

  public void removeListener(ARTListener al)
  { multicaster.removeListener(al); }
}
