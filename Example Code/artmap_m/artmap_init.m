function [artmap_net] = artmap_init( MAPTYPEArg, MArg, LArg,...
				     MAX_F2_SIZEArg,...
				     defaultParams, varargin )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_init.m
%
% Description: initialization for a single ARTMAP network
%
% Authors: Suhas Chelian, Norbert Kopco
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs and Default Values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAPTYPEArg = (nothing)
%   1 - Fuzzy ARTMAP
%   2 - ART-EMAP
%   3 - ARTMAP_IC
%   4 - DIST_ARTMAP
% MArg = (nothing)
%   The number of input dimensions, not including complement coding.
% L = (nothing)
%   The number of output classes, labeled 1 to L.
% MAX_F2_SIZE = (nothing)
%   The maximum number of F2 nodes.
% defaultParams = (nothing)
%   Whether to use default parameters or not.
%
% If defaultParams is set to one:
%   alpha = .01;
%   p = 1;
%   beta = 1;
%   epsilon = -.001;
%   rho_a_bar = 0;
%   rho_ab = .99;
%   Tu = alpha*M;
% Else (using 'varagin', set at either the first 6 or all of them)
%   alpha = varargin{1}; 
%   p = varargin{2};
%   beta = varargin{3};
%   epsilon = varargin{4};
%   rho_a_bar = varargin{5};
%   rho_ab = varargin{6};
%   Tu = varargin{7};
%
%   DO_WEBER = varagin{8};
%   DO_CBD = varagin{9};
%   DO_TRAIN_WTA = varagin{10};
%   DO_TRAIN_ICG = varagin{11};
%   DO_TRAIN_IC = varagin{12};
%   DO_TEST_IC = varagin{13};
%   DO_TEST_WTA = varagin{14};
%   DO_TEST_SCG = varagin{15};
%   DO_TEST_ICG = varagin{16};
%   DO_OLD_Tj = varargin{17};
%   DO_KAPPA_VEC = varargin{18};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_net
%   The initialized ARTMAP network.  artmap_net.fail is set to one
%   if a problem was detected in the initialization.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% /*****************************************************************************
%  * Constants and Parameters
%  ****************************************************************************/
%// ARTMAP type.  Toggle EXACTLY one flag.
MAPTYPE = MAPTYPEArg;
FUZZY_ARTMAP = 0;
ARTMAP_IC = 0;
ART_EMAP = 0;
DIST_ARTMAP = 0;

if ( MAPTYPE == 1 )
  FUZZY_ARTMAP = 1;
elseif ( MAPTYPE == 2 )
  ART_EMAP = 1;
elseif ( MAPTYPE == 3 )  
  ARTMAP_IC = 1;
else
  DIST_ARTMAP = 1;
end

%// Constants definitions and training/testing Data
M = MArg;		%// Dimensionality of input vectors (not including complement 
		%//	coding)
L = LArg;		%// Number of output classes
%EPOCHS = EPOCHSArg;	%// Number of training epochs
MAX_F2_SIZE = MAX_F2_SIZEArg;	%// Max number of F2 nodes.  Increase this if you run out
			%//	in training.

%// Parameters
if ( defaultParams == 1 )
 alphaArg = .01;	%// CBD and Weber law signal parameter
 pArg = 1;		%// CAM rule power
 betaArg = 1;		%// learning rate
 epsilonArg = -.001;	%// MT rule parameter
 rho_a_barArg = 0;	%// ARTa baseline vigilance
 rho_abArg = .99;       %// Mapfield vigilance
 TuArg = alphaArg*MArg;	%// F0 -> F2 signal to uncomited nodes.
			%//	NOTE:  Changes for each input with Weber
			%//	Law
else

 alphaArg = varargin{1}{1}; 
 pArg = varargin{1}{2};
 betaArg = varargin{1}{3};
 epsilonArg = varargin{1}{4};
 rho_a_barArg = varargin{1}{5};
 rho_abArg = varargin{1}{6};
 TuArg = varargin{1}{7};
 
end

alpha = alphaArg;		%// CBD and Weber law signal parameter
p = pArg;			%// CAM rule power
beta = betaArg;		%// learning rate
epsilon = epsilonArg;	%// MT rule parameter
rho_a_bar = rho_a_barArg;		%// ARTa baseline vigilance
rho_ab = rho_abArg;
Tu = TuArg;			%// F0 -> F2 signal to uncomited nodes.
			%//	NOTE:  Changes for each input with Weber
			%//	Law
						

F0_SIZE = M*2;	%// Size of the F0 layer. Identical to size of the F1 layer

F2_SIZE = MAX_F2_SIZE; %// Size of the F0 layer. Identical to size of
                       %//	the F1 layer

% /*****************************************************************************
%  * System Setup
%  ****************************************************************************/
%// Singal Rule, Weber or CBD
DO_WEBER = 0;
DO_CBD = FUZZY_ARTMAP | ART_EMAP | ARTMAP_IC | DIST_ARTMAP;
if ( DO_WEBER == 1 )
  DO_CBD = 0;
end

% // Training mode.  ICG = Increased CAM Gradient
DO_TRAIN_WTA = FUZZY_ARTMAP | ART_EMAP | ARTMAP_IC;
DO_TRAIN_ICG = DIST_ARTMAP;

% // Instance couting
DO_TRAIN_IC = DIST_ARTMAP | ARTMAP_IC;
DO_TEST_IC = DIST_ARTMAP | ARTMAP_IC;

% // Testing mode.  SCG = Simple CAM Gradient
DO_TEST_WTA = FUZZY_ARTMAP;
DO_TEST_SCG = 0;
DO_TEST_ICG = DIST_ARTMAP | ARTMAP_IC | ART_EMAP;

i = 0;
if ( FUZZY_ARTMAP )
  i = i + 1;
end
if ( ART_EMAP )
  i = i + 1;
end
if ( ARTMAP_IC )
  i = i + 1;
end
if ( DIST_ARTMAP )
  i = i + 1;
end

fail = 0;
if ( i ~=1 )
  disp( 'Wrong number of systems chosen.  Choose exactly one system!' ...
	);
  fail = 1;
%  quit;
end

if ( DIST_ARTMAP & ~DO_CBD )
  disp( 'dARTMAP must use CBD (check DO_CBD)' );
  fail = 1;
%  quit;
end

% Initialization of LTM weights
tau_ij = zeros(F0_SIZE,MAX_F2_SIZE);
tau_ji = zeros(MAX_F2_SIZE,F0_SIZE);
c = zeros(1,MAX_F2_SIZE);
% Initialize Kappa after we know what DO_KAPPA_VEC is

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pack
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

artmap_net.type = MAPTYPE;
artmap_net.M = M;
artmap_net.L = L;
artmap_net.EPOCHS = 0;              % (actually set in artmap_train)
artmap_net.MAX_F2_SIZE = MAX_F2_SIZE;

artmap_net.TRAIN_N = 0;             % (actually set in artmap_train)
artmap_net.input = [];              % "
artmap_net.output = [];             % "
artmap_net.TEST_N = 0;              % (actually set in artmap_test)
artmap_net.te_input = [];           % "
artmap_net.te_output = [];          % "

artmap_net.alpha = alpha;
artmap_net.p = p;
artmap_net.beta = beta;
artmap_net.epsilon = epsilon;
artmap_net.rho_a_bar = rho_a_bar;
artmap_net.rho_ab = rho_ab;
artmap_net.Tu = Tu;

artmap_net.F0_SIZE = M*2;
artmap_net.F2_SIZE = MAX_F2_SIZE;

if ( defaultParams | size(varargin{1},2) < 8 )
% Defaults
 artmap_net.DO_WEBER = DO_WEBER;
 artmap_net.DO_CBD = DO_CBD;
 artmap_net.DO_TRAIN_WTA = DO_TRAIN_WTA;
 artmap_net.DO_TRAIN_ICG = DO_TRAIN_ICG;
 artmap_net.DO_TRAIN_IC = DO_TRAIN_IC;
 artmap_net.DO_TEST_IC = DO_TEST_IC;
 artmap_net.DO_TEST_WTA = DO_TEST_WTA;
 artmap_net.DO_TEST_SCG = DO_TEST_SCG;
 artmap_net.DO_TEST_ICG = DO_TEST_ICG;
 
 artmap_net.DO_OLD_Tj = 0;
 artmap_net.DO_KAPPA_VEC = 0;
 kappa = zeros(1, MAX_F2_SIZE);
else
% On your own!
 artmap_net.DO_WEBER = varargin{1}{8};
 artmap_net.DO_CBD = varargin{1}{9};
 artmap_net.DO_TRAIN_WTA = varargin{1}{10};
 artmap_net.DO_TRAIN_ICG = varargin{1}{11};
 artmap_net.DO_TRAIN_IC = varargin{1}{12};
 artmap_net.DO_TEST_IC = varargin{1}{13};
 artmap_net.DO_TEST_WTA = varargin{1}{14};
 artmap_net.DO_TEST_SCG = varargin{1}{15};
 artmap_net.DO_TEST_ICG = varargin{1}{16};
 
 artmap_net.DO_OLD_Tj = varargin{1}{17};
 if ( artmap_net.DO_OLD_Tj )
   artmap_net.Tu = artmap_net.M;
 end
 
 artmap_net.DO_KAPPA_VEC = varargin{1}{18};
 if ( artmap_net.DO_KAPPA_VEC == 0 )
   kappa = zeros(1, MAX_F2_SIZE);
 else
   kappa = zeros( MAX_F2_SIZE, L );
 end
 
end

artmap_net.fail = fail;

artmap_net.tau_ij = tau_ij; % weights, buttom-up
artmap_net.tau_ji = tau_ji; % weights, top-down
artmap_net.c = c;           % instance counts
artmap_net.C = 0;           % number of commited nodes (actually
                            % set in artmap_train)
artmap_net.kappa = kappa;      % Mapfield (actually set in artmap_train)
artmap_net.Delta = [];      % refractory nodes (set in artmap_train)
artmap_net.rho = 0;
