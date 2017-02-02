package kcdc.nn.art;

import kcdc.nn.*;

public interface MapListener extends NNListener
{
  public void mapMismatch(MapEvent e);
  public void mapResize(MapEvent e);
  //public void mapTrained(MapEvent e);
}
