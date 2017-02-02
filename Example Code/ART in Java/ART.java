package kcdc.nn.art;

import java.util.*;

import kcdc.nn.*;

abstract public class ART extends AbstractNeuralNetwork implements ARTNetwork
{
  protected Vector neurons = new Vector();
  protected double learning_rate = 1.0;
  protected double vigilance = 0.0;

  protected ARTNeuron winner = null;
  protected int winner_index = -1;
  protected double confidence = 0.0;
  protected boolean activated = false;
  protected Pattern input_pattern = null;
  protected Pattern output_pattern = null;

  protected int resets = 0;
  protected int resizes = 0;

  public ART()
  {
    learning_rate = 1.0;
    vigilance = 0.0;
    size = 0;
    winner = null;
    winner_index = -1;
    resets = 0;
    resizes = 0;
    done = false;
  }

  public ART(int num_inputs, int input_style)
  {
    super(num_inputs, input_style);
    learning_rate = 1.0;
    vigilance = 0.0;
    size = 0;
    winner = null;
    winner_index = -1;
    resets = 0;
    resizes = 0;
    done = false;
  }

  public void epoch()
  {
    epochs++;
    done = true;
    resets = 0;
    resizes = 0;
  }

  public boolean train(PatternSet pattern_set, int max_epochs)
  {
    Pattern pattern = null;

    for (int e = 0; !done && (e < max_epochs); e++)
    {
      epoch();

      for (int i = 0; i < pattern_set.getNumPatterns(); i++)
      {
	pattern = pattern_set.getPattern(i);
	if (pattern instanceof DoublePattern)
	  train(((DoublePattern)pattern).getInputPattern());
	else train(pattern);
      }
    }

    return done;
  }

  /** Train ART network on inputs. */
  public boolean train(Pattern pattern)
  {
    setInputPattern(pattern);

    // If there are no neurons then add a new winner
    if (neurons.size() == 0)
    {
      addWinner(pattern);
      return false;
    }

    activate(pattern);

    int best_index = 0;
    ARTNeuron best = null;

    for (int i = 0; i < neurons.size(); i++)
    {
      best_index = getBestMatch();
      best = getNeuron(best_index);
      
      confidence = best.input/pattern.getSumInputs();

      // Test to see if the winning category meets vigilance requirement
      if (confidence >= vigilance)
      {
	// Found a match
	winner = best;
	winner_index = best_index;
	categorize(best);
	best.learn(pattern, learning_rate);
	return true;
      }
      else
      {
	// Failing vigilance requirement
	best.eligible = false;
	resets++;
	done = false;
	multicaster.artReset(new ARTEvent(this, ARTEvent.ART_RESET));
      }
    }
    
    // No eligible categories: add a winner
    addWinner(pattern);
    confidence = 1.0;
    activated = false;
    return false;
  }

  public boolean test(PatternSet pattern_set)
  {
    Pattern pattern = null;

    test_correct = 0;
    test_errors = 0;
    test_unanswered = 0;

    boolean correct = false;
    for (int i = 0; i < pattern_set.getSize(); i++)
    {
      pattern = pattern_set.getPattern(i);
      if (pattern instanceof DoublePattern)
	test(((DoublePattern)pattern).getInputPattern());
      else test(pattern);
    }

    return ((test_errors + test_unanswered) == 0);
  }

  public boolean test(Pattern pattern)
  {
    setInputPattern(pattern);
    winner = null;
    winner_index = -1;

    if (neurons.size() == 0)
    {
      test_unanswered++;
      return false;
    }

    activate(pattern);

    int best_index = 0;
    ARTNeuron best = null;

    for (int i = 0; i < neurons.size(); i++)
    {
      best_index = getBestMatch();
      best = getNeuron(best_index);
      // TODO: Is this a bug?
      winner = best;
      winner_index = best_index;

      confidence = best.input/pattern.getSumInputs();

      if (confidence >= vigilance)
      {
	winner = best;
	winner_index = best_index;
	categorize(best);
	test_correct++;
	return true;
      }
      else best.eligible = false;
    }

    test_unanswered++;
    return false;
  }

  public void activate(Pattern pattern)
  {
    ARTNeuron neuron = null;

    for (int i = 0; i < neurons.size(); i++)
    {
      neuron = (ARTNeuron) neurons.elementAt(i);
      neuron.eligible = true;
      neuron.activate(pattern);
    }

    activated = true;
  }

  public void categorize(ARTNeuron winner)
  {
    ARTNeuron neuron = null;

    // Winner-takes-all categories
    winner.output = 1;
    for (int i = 0; i < neurons.size(); i++)
    {
      neuron = (ARTNeuron) neurons.elementAt(i);
      if (neuron != winner) neuron.output = 0;
    }
  }

  public Pattern testReverse(Pattern outputs)
  {
    Pattern inputs = new Pattern();
    ARTNeuron neuron = null;

    for (int i = 0; i < neurons.size(); i++)
    {
      neuron = (ARTNeuron) neurons.elementAt(i);
      neuron.activateReverse(outputs.getInput(i));
      inputs.addInput(neuron.getInput());
    }

    return inputs;
  }

  public int getBestMatch()
  {
    if (!activated) return -1;

    // Find the most activated category
    ARTNeuron best = null;
    int best_index = -1;
    double best_output = -1.0;

    ARTNeuron neuron = null;

    for (int i = 0; i < neurons.size(); i++)
    {
      neuron = (ARTNeuron) neurons.elementAt(i);
      if (neuron.eligible && (neuron.output > best_output))
      {
	best = neuron;
	best_index = i;
	best_output = neuron.output;
      }
    }

    return best_index;
  }

  public double getLearningRate()
  { return learning_rate; }

  public void setLearningRate(double r)
  { learning_rate = r; }

  public double getVigilance()
  { return vigilance; }

  public void setVigilance(double v)
  { vigilance = v; }

  public Vector getNeurons()
  { return neurons; }

  public ARTNeuron getNeuron(int i)
  { return (ARTNeuron)neurons.elementAt(i); }

  public int getNumNeurons()
  { return neurons.size(); }

  public double getOutput(int i)
  { return ((ARTNeuron)neurons.elementAt(i)).output; }

  public Pattern getOutputPattern()
  {
    output_pattern = new Pattern();
    Enumeration e = neurons.elements();
    while(e.hasMoreElements())
      output_pattern.addInput(((ARTNeuron)e.nextElement()).output);
    output_pattern.setSource(this);
    return output_pattern;
  }

  public ARTNeuron getWinner()
  { return winner; }

  public int getWinnerIndex()
  { return winner_index; }

  public double getConfidence()
  { return confidence; }

  abstract public void addWinner(Pattern pattern);

  public int getNumResets()
  { return resets; }

  public int getNumResizes()
  { return resizes; }

  public int getNumEpochs()
  { return epochs; }

  public void setInputPattern(Pattern p)
  { input_pattern = p; }

  public Pattern getInputPattern()
  { return input_pattern; }

  //// Event processing ////

  protected ARTEventMulticaster multicaster = new ARTEventMulticaster();

  public void addListener(ARTListener al)
  { multicaster.addListener(al); }

  public void removeListener(ARTListener al)
  { multicaster.removeListener(al); }
}
