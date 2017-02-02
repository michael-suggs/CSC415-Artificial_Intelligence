package kcdc.nn.art;

import kcdc.nn.*;

public interface ARTListener extends NNListener
{
  public void artReset(ARTEvent e);
  public void artResize(ARTEvent e);
}
