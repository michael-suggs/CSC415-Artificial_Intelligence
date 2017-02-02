package kcdc.nn.art;

import java.util.*;

import kcdc.nn.*;

public class ARTEventMulticaster implements ARTListener
{
  Vector listeners;

  public ARTEventMulticaster()
  {
    listeners = new Vector();
  }

  public void addListener(ARTListener l)
  {
    listeners.addElement(l);
  }
  
  public void removeListener(ARTListener l)
  {
    listeners.removeElement(l);
  }
  
  public void artReset(ARTEvent event)
  {
    Enumeration e = listeners.elements();
    while (e.hasMoreElements())
      ((ARTListener) e.nextElement()).artReset(event);
  }

  public void artResize(ARTEvent event)
  {
    Enumeration e = listeners.elements();
    while (e.hasMoreElements())
      ((ARTListener) e.nextElement()).artResize(event);
  }
}
