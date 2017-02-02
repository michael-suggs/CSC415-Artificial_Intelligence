package kcdc.nn.art;

import java.util.*;

import kcdc.nn.*;

public class MapEvent extends NNEvent
{
  public static final int MAP_MISMATCH = 0x001;
  public static final int MAP_RESIZE = 0x002;
  public static final int MAP_TRAINED = 0x004;

  private int id;

  public MapEvent(Object source, int id)
  {
    super(source);
    this.id = id;
  }

  public int getID()
  { return id; }

  public String toString()
  { return new String("MapEvent( " + source + ")"); }
}
