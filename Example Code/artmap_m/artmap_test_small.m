function [Y] = artmap_test_small( artmap_net, A )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_test_small.m
%
% Description: testing an ARTMAP network for a single input
%
% Authors: Suhas Chelian, Norbert Kopco
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs and Default Values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_net = (nothing)
%   The artmap_net to train.
% A = (nothing)
%   The input, complement coded (size = 2M by 1).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Y
%   The F3 activation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Unpack
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MAPTYPE = artmap_net.type;
M = artmap_net.M;
L = artmap_net.L;
EPOCHS = artmap_net.EPOCHS;
MAX_F2_SIZE = artmap_net.MAX_F2_SIZE;

TRAIN_N = artmap_net.TRAIN_N;
input = artmap_net.input;
output = artmap_net.output;
TEST_N = artmap_net.TEST_N;
te_input = artmap_net.te_input;
te_output = artmap_net.te_output;

alpha = artmap_net.alpha;
p = artmap_net.p;
beta = artmap_net.beta;
epsilon = artmap_net.epsilon;
rho_a_bar = artmap_net.rho_a_bar;
rho_ab = artmap_net.rho_ab;
Tu = artmap_net.Tu;

F0_SIZE = artmap_net.F0_SIZE;
F2_SIZE = artmap_net.F2_SIZE;

DO_WEBER = artmap_net.DO_WEBER;
DO_CBD = artmap_net.DO_CBD;
DO_TRAIN_WTA = artmap_net.DO_TRAIN_WTA;
DO_TRAIN_ICG = artmap_net.DO_TRAIN_ICG;
DO_TRAIN_IC = artmap_net.DO_TRAIN_IC;
DO_TEST_IC = artmap_net.DO_TEST_IC;
DO_TEST_WTA = artmap_net.DO_TEST_WTA;
DO_TEST_SCG = artmap_net.DO_TEST_SCG;
DO_TEST_ICG = artmap_net.DO_TEST_ICG;

DO_OLD_Tj = artmap_net.DO_OLD_Tj;
DO_KAPPA_VEC = artmap_net.DO_KAPPA_VEC;

tau_ij = artmap_net.tau_ij;
tau_ji = artmap_net.tau_ji;
c = artmap_net.c;
C = artmap_net.C;
kappa = artmap_net.kappa;


  %Step 2 - Reset
  S = sum(min(A*ones(1,C),(1-tau_ij(:,1:C))));
  
  % Note that Theta is constant throughout testing so it need not be
  % recomputed for each input
  if ( DO_OLD_Tj )
    Theta = sum(tau_ij(:,1:C));
  else
    Theta = sum(tau_ij(:,1:C)) - M;
  end
  
  if ( DO_CBD )
    T(1,1:C) =  S + (1-alpha)*Theta;
  else
    T(1,1:C) = S ./ (alpha + sum(1-tau_ij(:,1:C)));
    Tu = M/(alpha + 2*M);
  end

  Lambda = find(T>=Tu);
  y = zeros(1,C);
  Y = zeros(1,C);

  if ( DO_TEST_ICG )
    %// (a) If the network is in distributed mode: F2 nodes are activated
    %//     according to the increased gradient CAM rule.
	
    if ( DO_OLD_Tj )
      alMT = (2-alpha)*M-T;
    else
      alMT = (M-T);
    end
        
    % (i) if M - T >0 for all j belonging to Lambda
    if(all(alMT(Lambda)>0))
      for j=1:length(Lambda)
	y(Lambda(j)) = 1/(1+ ...
			  sum(((Lambda~=Lambda(j))*alMT(Lambda(j))./ alMT(Lambda)).^p'));
      end
    else
      % (ii) Point box case
      Lambda_pp = find(alMT(Lambda)==0);
      y(Lambda_pp)=1/length(Lambda_pp);
    end
    
    %// F3 activation
    if ( DO_TEST_IC )
      Y(1:C) = c(1:C).*y(1:C)/sum(c(1:C).*y(1:C));
    else
      Y(1:C) = y(1:C) / sum(y(1:C));
    end
  end

  if ( DO_TEST_SCG )
    %// F2 activation
    y(1:C) = T.^p/(sum(T.^p));
    
    %// F3 activation
    if ( DO_TEST_IC )
      Y(1:C) = c(1:C).*y(1:C)/sum(c(1:C).*y(1:C));
    else
      Y(1:C) = y(1:C);
    end
  end
  
  %// (b) If the network is in WTA mode: Only one F2 node, with j=J, is activated
  if ( DO_TEST_WTA )
    J = find(T == max(T));
    if ( size(J,2) > 1 )
      J = J(1);
    end
    
    %// (ii) F2 and F3 activation  
    y = ([1:C] == J);
    Y = y;
  end
