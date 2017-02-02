%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_shell.m
%
% Description: a sample program of how to use a single ARTMAP network
%
% Authors: Suhas Chelian, Norbert Kopco
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%clear all;

traceInit = 0; % Toggle this if you want to see how the network was
               % initialized
traceTrain = 0;% Toggle this if you want to see what weights the
               % network developed after training

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data

% These are the points given in Fig. 10 of the dARTMAP paper
% (Carpenter et al, 1998).  You can check the weights by toggling
% "traceTrain".
input = [.8,.5; .5,.2; .8,.8; .7,.1; 1,1; 1,1; .6,.4; .2,.3];
output = [2; 2; 1; 1; 1; 1; 2; 2];

TEST_N = 8;	%// Number of testing points
te_input = [.2,.9; .9,.6; .6,.6; .9,.8; .7,.5; .2,.7; .4,.9; .9,.7];
te_output = [2; 2; 1; 2; 1; 2; 2; 2];

% Uncomment these lines to use the large training sets
% load input.dat;
% load output.dat;

train = [input, output];
trainN = size(input,1);
%trainN = 5;

% Uncomment these lines to use the large testing sets
% load te_input.dat;
% load te_output.dat;

test = [te_input, te_output];
testN = size(te_input,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Init
disp( 'Initializing' );
MAPTYPE = 4;
M = 2;
L = 2;
MAX_F2_SIZE = 100;
defaultParams = 1;

myVarargin = cell(1,6);
myVarargin{1} = .01;			% alpha
myVarargin{2} = 1;			% p
myVarargin{3} = 1;			% beta
myVarargin{4} = -.001;			% epsilon
myVarargin{5} = 0;			% rho_a_bar
myVarargin{6} = .99;                    % rho_ab
myVarargin{7} = myVarargin{1}*M;	% Tu

% Uncomment these lines to configure even more
% myVarargin{8} = 0; % Weber
% myVarargin{9} = 1; % CBD
% myVarargin{10} = 0; % Train WTA
% myVarargin{11} = 1; % Train ICG
% myVarargin{12} = 1; % Train IC
% myVarargin{13} = 1; % Test IC
% myVarargin{14} = 0; % Test WTA
% myVarargin{15} = 0; % Test SCG
% myVarargin{16} = 1; % Test ICG
% myVarargin{17} = 0; % DO_OLD_Tj
% myVarargin{18} = 1; % DO_KAPPA_VEC

if ( defaultParams == 0 )
	artmap_net = artmap_init( MAPTYPE, M, L, MAX_F2_SIZE,...
			  defaultParams, myVarargin );
else
	artmap_net = artmap_init( MAPTYPE, M, L, MAX_F2_SIZE,...
			  defaultParams );
end

if ( artmap_net.fail == 1 )
  disp( 'artmap_init failed!' );
  quit;
end

if ( traceInit )
  artmap_net
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train
forceInputHC = 0;
forceOutputHC = 0;
verbose = 1;
defaultEpochs = 1;

myVarargin2 = cell(1,3);
myVarargin2{1} = 3;			% EPOCHS
myVarargin2{2} = 1;			% shuffle
myVarargin2{3} = 0;			% shuffleSeed

if ( defaultEpochs == 1 )
  [artmap_net] = artmap_train_large( artmap_net, train, trainN,...
				   forceInputHC, forceOutputHC, ...
				   verbose, defaultEpochs );
else
  [artmap_net] = artmap_train_large( artmap_net, train, trainN,...
				   forceInputHC, forceOutputHC, ...
				   verbose, defaultEpochs, myVarargin2 ...
				     );
end

if ( traceTrain )
%  Uncomment this line to see the entire network
%  artmap_net

  disp( 'C' );
  artmap_net.C
  disp( 'tau_ij' );
  artmap_net.tau_ij(:, 1:artmap_net.C)
  disp( 'tau_ji' );
  artmap_net.tau_ji(1:artmap_net.C, :)
  disp( 'c' );
  artmap_net.c(1:artmap_net.C)
  disp( 'kappa' );
  if ( artmap_net.DO_KAPPA_VEC == 0 )
    artmap_net.kappa(1:artmap_net.C)
  else
    artmap_net.kappa(1:artmap_net.C,:)
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test
forceInputHC = 0;
forceOutputHC = 0;
 % Verbose level of 3 only good for small testing sets
if ( testN <= 10 )
  verbose = 3;
else
  verbose = 2;
end


[artmap_net, pc, correct, bigSigma, bigY] = artmap_test_large( artmap_net, test, testN,...
				   forceInputHC, forceOutputHC, verbose );

disp( sprintf( 'Percent Correct: %3.2f%%', pc*100 ) );
