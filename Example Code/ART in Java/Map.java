package kcdc.nn.art;

import java.util.*;

import kcdc.nn.*;

public class Map extends AbstractNeuralNetwork implements ARTListener, MapListener
{
  public Map()
  {
  }

  public String toString()
  {
    return new String("Map(" + vigilance + ", " + neurons + ")");
  }

  private void addNeuron(Pattern input)
  {
    MapNeuron neuron = null;

    // Add winning neuron
    if (input != null)
    {
      neuron = new MapNeuron(input.getNumInputs());
      neuron.activate(input);
    }
    else
    {
      neuron = new MapNeuron(0);
    }

    neurons.addElement(neuron);

    resizes++;
    size = neurons.size();
    multicaster.mapResize(new MapEvent(this, MapEvent.MAP_RESIZE));
  }

  private void addInput(Pattern input, Pattern output)
  {
    MapNeuron neuron = null;

    int i = 0;
    Enumeration e = neurons.elements();
    while (e.hasMoreElements())
    {
      neuron = (MapNeuron)e.nextElement();
      neuron.addWeight(output.getInput(i++));
      neuron.activate(input);
    }

    num_inputs++;
  }

  public void activate(Pattern input)
  {
    // Calculate Map activations
    Enumeration e = neurons.elements();
    while(e.hasMoreElements())
      ((MapNeuron)e.nextElement()).activate(input);
  }

  public boolean test(PatternSet patterns)
  {
    done = true;
    for (int i = 0; i < patterns.getSize(); i++)
      done &= test(patterns.getPattern(i));
    
    return done;
  }

  private DoublePattern test_pattern = null;

  public boolean test(Pattern pattern)
  {
    if (!(pattern instanceof DoublePattern))
      return false; // NNException

    test_pattern = (DoublePattern) pattern;
    setInputPattern(test_pattern.getInputPattern());
    setOutputPattern(test_pattern.getOutputPattern());

    // Activate Map
    activate(input_pattern);
    // Calculate the intersection and union of response and the outputs
    double intersection = Fuzzy.intersect(getResponsePattern(),output_pattern);
    //double union = Fuzzy.union(activations, outputs);
    double union = 1.0;
    confidence = intersection/union;

    // Check Map vigilance
    if (confidence < getVigilance())
    {
      // If there is a mismatch broadcast a message to all listeners
      mismatches++;
      multicaster.mapMismatch(mismatch_event);
      return false;
    }
    else
    {
      // Return true if no mismatches (i.e. successful test)
      return true;
    }
  }

  public void epoch()
  {
    mismatches = 0;
  }

  public boolean train(PatternSet patterns, int max_epoch)
  {
    for (int i = 0; !done && (i < max_epoch); i++)
    {
      epoch();
      done = true;
      for (int j = 0; i < patterns.getSize(); j++)
	done &= train(patterns.getPattern(j));
    }

    return done;
  }

  private DoublePattern train_pattern = null;

  public boolean train(Pattern pattern)
  {
    if (!(pattern instanceof DoublePattern))
      return false; // NNException

    train_pattern = (DoublePattern) pattern;
    setInputPattern(train_pattern.getInputPattern());
    setOutputPattern(train_pattern.getOutputPattern());

    // Test the map for mismatches
    if (test(train_pattern))
    {
      // If there is was no mismatch then update neuron weights
      int winner = ((ART)input_pattern.getSource()).getWinnerIndex();
      for (int i = 0; i < output_pattern.getNumInputs(); i++)
	getNeuron(i).setWeight(winner, output_pattern.getInput(i));

      // Broadcast a message to all listeners that map has been trained
      //multicaster.mapTrained(new MapEvent(this, MapEvent.MAP_TRAINED));
      // Return true. The map was successfully trained 
      return true;
    }
    else
    {
      //mismatches++;
      // If there is a mismatch broadcast a message to all listeners
      //multicaster.mapMismatch(mismatch_event);
      //if (((ART)input_source).getVigilance() == 1.0)
      // Return true to halt training if vigilance is at maximum
      //return true;
      // Return false. The map could not be trained because of mismatch
      return false;
    }
  }

  public void artReset(ARTEvent e)
  {
  }

  /** The artResize event handler catches all resize events produced
    * by either the ARTa or ARTb networks. Depending on which ART
    * network was resized an different action is taken to update the
    * map.
    *
    * <ul>
    * <li>If input is resized then the weights are increased to match
    * <li>If output is resized then the neurons are increased to match
    * </ul>
    *
    */
  public void artResize(ARTEvent e)
  {
    setInputPattern(((ARTNetwork)input_source).getOutputPattern());
    setOutputPattern(((ARTNetwork)output_source).getOutputPattern());

    if (e.getSource() == input_source)
      // Expand the weights for each neuron to match ARTa
      addInput(input_pattern, output_pattern);
    else if (e.getSource() == output_source)
      // Increase the number of neurons in the map to match ARTb
      while (neurons.size() < output_pattern.getNumInputs())
	addNeuron(input_pattern);
  }

  protected MapEvent mismatch_event = new MapEvent(this,MapEvent.MAP_MISMATCH);
  protected MapEvent trained_event = new MapEvent(this,MapEvent.MAP_TRAINED);

  //// Access methods ////

  public double getVigilance()
  { return vigilance; }

  public void setVigilance(double v)
  { vigilance = v; }

  public double getConfidence()
  { return confidence; }

  public MapNeuron getNeuron(int i)
  { return (MapNeuron)neurons.elementAt(i); }

  public double getOutput(int i)
  { return ((MapNeuron)neurons.elementAt(i)).getOutput(); }

  public void setInputPattern(Pattern pattern)
  { input_pattern = pattern; }

  public Pattern getInputPattern()
  { return input_pattern; }

  public void setOutputPattern(Pattern pattern)
  { output_pattern = pattern; }

  public Pattern getOutputPattern()
  { return output_pattern; }

  public Pattern getResponsePattern()
  {
    Pattern response_pattern = new Pattern();
    Enumeration e = neurons.elements();
    while (e.hasMoreElements())
      response_pattern.addInput(((MapNeuron)e.nextElement()).getOutput());
    return response_pattern;
  }

  public int getNumMismatches()
  { return mismatches; }

  public int getNumResizes()
  { return resizes; }

  protected double vigilance = 1.0;
  protected double confidence = 0.0;
  protected Vector neurons = new Vector();
  protected Pattern input_pattern = null;
  protected Pattern output_pattern = null;

  public void setInputSource(Object o)
  { input_source = o; }

  public void setOutputSource(Object o)
  { output_source = o; }

  protected Object input_source = null;
  protected Object output_source = null;

  protected int mismatches = 0;
  protected int resizes = 0;
  protected boolean initialized = false;

  // Event processing
  MapEventMulticaster multicaster = new MapEventMulticaster();

  public void addListener(MapListener l)
  { multicaster.addListener(l); }

  public void removeListener(MapListener l)
  { multicaster.removeListener(l); }

  public void mapMismatch(MapEvent e)
  {
  }

  public void mapResize(MapEvent e)
  {
  }
}
