function [artmap_net] = artmap_train_large( artmap_net, trainArg, trainNArg,...
					    forceInputHC, forceOutputHC,...
					    verbose,...
					    defaultEpochs, varargin )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_train_large.m
%
% Description: training an ARTMAP network on a set
%
% Authors: Suhas Chelian, Norbert Kopco
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs and Default Values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_net = (nothing)
%   The artmap_net to train.
% trainArg = (nothing)
%   The training points.  The first 1:M columns are the inputs, and
%   the M+1st column is the output.
% trainNArg = (nothing)
%   The number of training points.
% forceInputHC = (nothing)
%   Whether to force input hypercubing or not.
% forceOutputHC = (nothing)
%   Whether to force proper indexing of output classes or not.
% verbose = (nothing)
%   A value of 1 will show the beginning and ending of training; a
%   value of 2 will show progress incrementally.
% defaultEpochs = (nothing)
%   Whether to use default training regime or not.
%
% If defaultEpochs is set to one:
%   EPOCHS = 1;
%   shuffle = 0;
%   shuffleSeed = 0;
% Else (using 'varargin')
%   EPOCHS = varargin{1};
%   shuffle = varargin{2};
%   shuffleSeed = varargin{3};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_net
%   The armtap_net after training.
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

fail = artmap_net.fail;

tau_ij = artmap_net.tau_ij;
tau_ji = artmap_net.tau_ji;
c = artmap_net.c;
C = artmap_net.C;
kappa = artmap_net.kappa;
Delta = artmap_net.Delta;
rho = artmap_net.rho;

if ( defaultEpochs == 1 )
  EPOCHS = 1;
  shuffle = 0;
  shuffleSeed = 0;
else
  EPOCHS = varargin{1}{1};
  shuffle = varargin{1}{2};
  shuffleSeed = varargin{1}{3};
end

if ( shuffle == 1 )
  rand( 'state', shuffleSeed ); % Seed "rand."  This effects
                                % "randperm" and hence the training
                                % order(s)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TRAIN_N = trainNArg;
input = trainArg(:,1:M);
output = trainArg(:,M+1);

%// Force input hypercubing
if ( forceInputHC == 1 )
 minM = min(input(:,1:M));  
 rangeM = max(input(:,1:M)) - minM;

 if ( TRAIN_N ~= 1 )
   for i = 1:M
     if (rangeM(i) == 0 )
       if ( max(input(:,i)) == 0 )
	 ; % do nothing
       else
	 input(:,i) = (input(:,i)-minM(i))/max(input(:,i));
       end
     else
       input(:,i) = (input(:,i)-minM(i))/rangeM(i);
     end
   end
 else
   if (rangeM == 0)
     if ( max(input) == 0 )
       ; % do nothing
     else
       input = (input-minM)/max(input);
     end
   else
     input = (input-minM)/rangeM;
   end
 end    

end

%// Check unit hypercubing
minM = min(input(:,1:M));
maxM = max(input(:,1:M));
findminM = find( minM < 0 );
findmaxM = find( maxM > 1 );
if ( ~isempty(findminM) | ~isempty(findmaxM) )  
  disp( 'Training input is not in unit hypercube!' );
%  quit;
  fail = 1;
end

%// Force proper indexing of output classes
if ( forceOutputHC == 1 )
 findminL = find( output < 1 );
 findmaxL = find( output > L );
 if ( ~isempty(findminL) )
   output( findminL ) = output( findminL ) + (1-output(findminL));
 end
 if ( ~isempty(findmaxL) )
   output( findmaxL ) = output( findmaxL ) - (output(findmaxL)-L);
 end
end

%// Check output classes
minL = min(output);
maxL = max(output);
if ( minL < 1 | maxL > L )
  disp( 'Training output is not indexed properlly!' );
%  quit;
  fail = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ( verbose >= 1 )
  disp( 'Training' );
end

for epochs = 1:EPOCHS
  
  if ( shuffle == 1 )
    shuffleIdx = randperm( TRAIN_N );
    input = input( shuffleIdx, : );
    output = output( shuffleIdx, : );
  end
  
  mark = 0; % keeps track of percent done
  
  for n=1:TRAIN_N
    if ( verbose >= 3 )
      disp( sprintf( 'n = %d', n ) );
    end
    
    %// Copy the input pattern into the F1 layer with complement codin
    A = [input(n,:) 1-input(n,:)]';

    %// Copy the corresponding output class into variable 
    K = output(n);

    [artmap_net] = artmap_train_small( artmap_net, A, K, verbose, n );
    if ( verbose >= 2 & (n/TRAIN_N) > (mark+.1) )
      mark = n/TRAIN_N;
      disp( sprintf( '  Percent done: %5.2f%%', mark*100 ) );
    end
    

  end

  if ( verbose >= 1 )
    disp( sprintf( 'Epoch: %3d	Commited F2 nodes: %3d', epochs, artmap_net.C ...
		   ) );
    epochs = epochs + 1;
  end
  
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pack
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DO NOT PACK HERE BECAUSE THINGS ARE LOCALLY UNCHANGED!, except
% EPOCHS and input

%artmap_net.type = MAPTYPE;
%artmap_net.M = M;
%artmap_net.L = L;
artmap_net.EPOCHS = EPOCHS;
%artmap_net.MAX_F2_SIZE = MAX_F2_SIZE;

artmap_net.TRAIN_N = TRAIN_N;
artmap_net.input = input;
artmap_net.output = output;
%artmap_net.TEST_N = TEST_N;
%artmap_net.te_input = te_input;
%artmap_net.te_output = te_output;

%artmap_net.alpha = alpha;
%artmap_net.p = p;
%artmap_net.beta = beta;
%artmap_net.epsilon = epsilon;
%artmap_net.rho_a_bar = rho_a_bar;
%artmap_net.Tu = Tu;

%artmap_net.F0_SIZE = M*2;
%artmap_net.F2_SIZE = MAX_F2_SIZE;

% artmap_net.DO_WEBER = DO_WEBER;
% artmap_net.DO_CBD = DO_CBD;
% artmap_net.DO_TRAIN_WTA = DO_TRAIN_WTA;
% artmap_net.DO_TRAIN_ICG = DO_TRAIN_ICG;
% artmap_net.DO_TRAIN_IC = DO_TRAIN_IC;
% artmap_net.DO_TEST_IC = DO_TEST_IC;
% artmap_net.DO_TEST_WTA = DO_TEST_WTA;
% artmap_net.DO_TEST_SCG = DO_TEST_SCG;
% artmap_net.DO_TEST_ICG = DO_TEST_ICG;

% artmap_net.DO_OLD_Tj = DO_OLD_Tj;
% artmap_net.DO_KAPPA_VEC = DO_KAPPA_VEC;

% artmap_net.fail = fail;

%artmap_net.tau_ij = tau_ij; % weights
%artmap_net.tau_ji = tau_ji;
%artmap_net.c = c;           % instance counts
%artmap_net.C = C;           % number of commited nodes
%artmap_net.kappa = kappa;      % Wab
%artmap_net.Delta = Delta;
%artmap_net.rho = rho;
