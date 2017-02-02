package kcdc.nn.art;

import java.util.*;

import kcdc.nn.*;

public class MapEventMulticaster implements MapListener
{
  Vector listeners;

  public MapEventMulticaster()
  {
    listeners = new Vector();
  }

  public void addListener(MapListener l)
  {
    listeners.addElement(l);
  }
  
  public void removeListener(MapListener l)
  {
    listeners.removeElement(l);
  }
  
  public void mapMismatch(MapEvent event)
  {
    Enumeration e = listeners.elements();
    while (e.hasMoreElements())
      ((MapListener) e.nextElement()).mapMismatch(event);
  }

  public void mapResize(MapEvent event)
  {
    Enumeration e = listeners.elements();
    while (e.hasMoreElements())
      ((MapListener) e.nextElement()).mapResize(event);
  }
}
