package kcdc.nn.art;

import java.util.*;

import kcdc.nn.*;

public class ARTMAP extends AbstractNeuralNetwork implements ARTListener, MapListener
{
  protected ARTNetwork ARTa = null;
  protected ARTNetwork ARTb = null;
  protected Map map = new Map();
  protected double base_vigilance = 0.0;
  protected boolean initialized = false;

  public ARTMAP()
  {
    // Create a new Map
    map = new Map();
    map.addListener(this);
  }

  public ARTMAP(ARTNetwork a, ARTNetwork b)
  {
    // Create a new Map
    map = new Map();
    map.addListener(this);

    if (a != null)
    {
      setNumInputs(a.getNumInputs());
      setInputStyle(a.getInputStyle());
    }

    if (b != null)
    {
      setNumOutputs(b.getNumInputs());
      setOutputStyle(b.getInputStyle());
    }

    setARTa(a);
    setARTb(b);
  }

  public void init()
  {
    super.init();

    // Create ARTa network for to match input patterns
    if (ARTa == null)
      if (getInputType() == Pattern.BINARY)
	setARTa(new ART1(getNumInputs(), getInputStyle()));
      else if (getInputType() == Pattern.ANALOG)
	setARTa(new FuzzyART(getNumInputs(), getInputStyle()));

    // Create ARTb networks for to match output patterns
    if (ARTb == null)
      if (getOutputType() == Pattern.BINARY)
	setARTb(new ART1(getNumOutputs(), getOutputStyle()));
      else if (getOutputType() == Pattern.ANALOG)
	setARTb(new FuzzyART(getNumOutputs(), getOutputStyle()));

    // If there was an error: exit
    if ((ARTa == null) || (ARTb == null))
    {
      System.err.println("Unknown input type.");
      System.exit(1);
      // NNException
    }

    // Initialize ART networks
    ARTa.init();
    ARTb.init();

    // Set initialized flag
    initialized = true;
  }

  public String toString()
  {
    StringBuffer buffer = new StringBuffer();
    buffer.append("ARTMAP(");
    buffer.append(ARTa);
    buffer.append(", ");
    buffer.append(ARTb);
    buffer.append(", ");
    buffer.append(map);
    buffer.append(")");
    return buffer.toString();
  }

  //// Methods ////

  public void epoch()
  {
    ARTa.epoch();
    ARTb.epoch();
    map.epoch();
    epochs++;
  }

  public boolean train(PatternSet pattern_set, int max_epochs)
  {
    done = false;
    for (int e = 0; (e < max_epochs) && !done; e++)
    {
      epoch();

      for (int i = 0; i < pattern_set.getSize(); i++)
	train(pattern_set.getPattern(i));
      done = (ARTa.isDone() && ARTb.isDone());
    }

    return done;
  }

  private DoublePattern train_pattern = null;
  private DoublePattern test_pattern = null;
  private DoublePattern map_pattern = new DoublePattern();

  public boolean train(Pattern pattern)
  {
    if (!(pattern instanceof DoublePattern))
      return false; // NNException

    train_pattern = (DoublePattern)pattern;

    // Train the ARTb network (reset events are caught by "artReset")
    boolean artb_trained = ARTb.train(train_pattern.getOutputPattern());
    map_pattern.setOutputPattern(ARTb.getOutputPattern());

    // Reset ARTa vigilance to base vigilance
    ARTa.setVigilance(base_vigilance);

    boolean done = false;
    while (!done)
    {
      // While ARTa can find a winner and the Map has a mismatch
      // train the Map to properly learn the mapping from ARTa
      // to ARTb. Training stops when either there are no more mismatches
      // or the vigilance is at it's maximum for ARTa.

      // Test the ARTa network to activate output
      if (ARTa.test(train_pattern.getInputPattern()))
      {
	map_pattern.setInputPattern(ARTa.getOutputPattern());

	// Successful test means there was no mismatch
	if (map.test(map_pattern))
	{
	  // Train the map
	  boolean map_trained = map.train(map_pattern);
	  // Train ARTa
	  boolean arta_trained = ARTa.train(train_pattern.getInputPattern());
	  // Return training success
	  done = true;
	}
	else
	{
	  // Map failed test increase ARTa vigilance
	  ARTa.setVigilance(Math.min(1.0, ARTa.getConfidence() + 0.0001));
	  // If vigilance is at maximum end training
	  if (ARTa.getVigilance() >= 1.0) done = true;
	}
      }
      else
      {
	// Resize events are handled by the Map automatically
	ARTa.train(train_pattern.getInputPattern());
	map_pattern.setInputPattern(ARTa.getOutputPattern());
	map.train(map_pattern);
	done = true;
      }
    }

    return true;
  }

  public boolean test(PatternSet patterns)
  {
    test_correct = 0;
    test_errors = 0;
    test_unanswered = 0;

    for (int i = 0; i < patterns.getSize(); i++)
      if (!test(patterns.getPattern(i)))
	test_errors++;
      else test_correct++;

    return ((test_errors + test_unanswered) == 0);
  }

  public boolean test(Pattern pattern)
  {
    if (!(pattern instanceof DoublePattern))
      return false; // NNException

    // Test the ART networks
    test_pattern = (DoublePattern) pattern;
    ARTb.test(test_pattern.getOutputPattern());
    ARTa.test(test_pattern.getInputPattern());
    // Return the success flag of the map
    map_pattern.setInputPattern(ARTa.getOutputPattern());
    map_pattern.setOutputPattern(ARTb.getOutputPattern());
    return map.test(map_pattern);
  }

  //// Event Handlers ////

  /** Mismatches cause increases in ARTa vigilance unless doing
    * so would exceed the maximum vigilance level. If this is
    * the case, then mismatch was caused by inconsistent or
    * noisy data so it should be ignored and training should
    * carry on.
    */
  public void mapMismatch(MapEvent e)
  {
//     if (e.getSource() == map)
//     {
//       ARTNeuron winner = ARTa.getWinner();
//       if (winner != null)
//       {
// 	double winner_input = winner.getInput();
// 	double sum_inputs = ARTa.getInputPattern().getSumInputs();
// 	ARTa.setVigilance(Math.min(1.0, (winner_input/sum_inputs) + 0.0001));
//       }
//     }
  }

  public void mapResize(MapEvent e)
  {
  }

  public void artReset(ARTEvent e)
  {
  }

  public void artResize(ARTEvent e)
  {
  }

  //// Access Methods ////

  public double getBaseVigilance()
  { return base_vigilance; }

  public void setBaseVigilance(double bv)
  { base_vigilance = bv; }

  public void setARTa(ARTNetwork a)
  { 
    if (a != null)
    {
      if (ARTa != null)
      {
	ARTa.removeListener(this);
	ARTa.removeListener(map);
      }

      // Check ART network matches input and output pattern information
      if ((a.getNumInputs() == getNumInputs()) &&
	  (a.getInputStyle() == getInputStyle()))
      {
	ARTa = a;
	ARTa.setVigilance(base_vigilance);
	ARTa.addListener(this);
	//ARTa.addListener(map.getInputListener());
	ARTa.addListener(map);
	map.setInputSource(ARTa);
	initialized = false;
      }
      else
	System.err.println("ARTa network does not match inputs. Ignoring.");
    }
  }

  public ARTNetwork getARTa()
  { return ARTa; }

  public void setARTb(ARTNetwork b)
  {
    if (b != null)
    {
      if (ARTb != null)
      {
	ARTb.removeListener(this);
	ARTb.removeListener(map);
      }

      // Check ART network matches input and output pattern information
      if ((b.getNumInputs() == getNumOutputs()) &&
	  (b.getInputStyle() == getOutputStyle()))
      {
	ARTb = b;
	ARTb.setVigilance(base_vigilance);
	ARTb.addListener(this);
	//ARTb.addListener(map.getOutputListener());
	ARTb.addListener(map);
	map.setOutputSource(ARTb);
	initialized = false;
      }
      else
	System.err.println("ARTb network does not match outputs. Ignoring.");
    }
  }

  public ARTNetwork getARTb()
  { return ARTb; }

  public Map getMap()
  { return map; }

  public int getNumMismatches()
  { return map.getNumMismatches(); }
}
