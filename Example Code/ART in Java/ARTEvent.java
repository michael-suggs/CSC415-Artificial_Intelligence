package kcdc.nn.art;

import java.util.*;

import kcdc.nn.*;

public class ARTEvent extends NNEvent
{
  public static final int ART_RESET = 0x001;
  public static final int ART_RESIZE = 0x002;

  private int id;

  public ARTEvent(Object source, int id)
  {
    super(source);
    this.id = id;
  }

  public int getID()
  { return id; }

  public String toString()
  { return new String("ARTEvent( " + source + ")"); }
}
