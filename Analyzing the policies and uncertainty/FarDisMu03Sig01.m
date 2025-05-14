

function FarDisMu03Sig01

% Preliminary tasks
demosetup(mfilename)


%% FORMULATION
  
% Model parameters
a = [18000 1 0.1 0.005 5 0];
b = [0 0 1237.56 -0.5];
nu =[18621.39 10];
%ita 1 was 3.20, but we added 0.05 of residue which is A base level of residue from PE (or BDM for that matter) whether or not the mulch goes to a landfill

ita = [3.25 0.5 0.1 3.2 -0.5];
delta = 0.9;
f = 0.04172;
p = 1.01;
z = 1;
sigma = 0.1;
mu = 0.3;
% Continuous state shock distribution
m = 4;                                 	% number of shocks
[e,w] = qnwnorm(m,mu,sigma^2) 	% shocks and probabilities

% Model structure
model.func = @func;                             % model functions
model.params = {a,b,nu,ita,z,f,p};              % function parameters
model.discount = delta;                         % discount factor
model.ds = 1;                                   % dimension of continuous state
model.dx = 1;                                   % dimension of continuous action
model.ni = 0;                                   % number of discrete states
model.nj = 0;                                   % number of discrete actions
model.e  = e;                           % continuous state shocks
model.w  = w;                           % continuous state shock probabilities

% Approximation structure
n    = 100;                                     	% number of collocation nodes
smin = 0;                                       	% minimum state
smax = 50;                                          % maximum state
basis = fundefn('spli',n,smin,smax);                % basis functions


%% SOLUTION
  
% Deterministic steady-state
qstar =max(0,((nu(2)+f*(-b(2)+b(4)*z)+a(2)*p)*(1-delta*ita(3))*(1-ita(3))+(a(6)...
    *(1-delta*ita(3))+delta*a(5)*(-ita(2)-ita(5)*z))*p*(ita(1)-ita(4)*z)+...
    a(3)*delta*(-ita(2)-ita(5)*z)*p*(1-ita(3)))/...
    ((-a(4)*(1-ita(3)*delta)+a(6)*delta*(ita(2)+ita(5)*z))*p*(1-ita(3))+(-a(6)*(1-delta*ita(3))...
    +(a(5)*delta*(ita(2)+ita(5)*z))*p*(-ita(2)-ita(5)*z)))); 	% Choice variable
sstar = (ita(1)-ita(2)*qstar-ita(4)*z-ita(5)*z*qstar)/(1-ita(3));           % State Variable
lstar = p*(-a(3)-a(5)*sstar-a(6)*qstar)/(-delta*ita(3)+1);                            % Shadow price for pollutant
pistar = (p*(a(1)-a(2)*qstar-a(3)*sstar-0.5*a(4)*qstar.^2-0.5*a(5)*sstar.^2-a(6)*qstar.*sstar)...
        -nu(1)-nu(2)*qstar-f*(b(1)-b(2)*qstar+b(3)*z+b(4)*z*qstar))/(1-delta);

% Check model derivatives
dpcheck(model,sstar,qstar);

% Solve collocation equation
[c,s,v,q,resid] = dpsolve(model,basis);


%% ANALYSIS
 
% Plot optimal policy
figure
hold on
plot(s,q,'color',[0,0,0])
title('Optimal Policy')
xlabel('pollutant stock')
ylabel('decay rate')

% Plot value function
figure
plot(s,v,'color',[0,0,0])
title('Value Function')
xlabel('pollutant stock')
ylabel('Lifetime profit')

% Plot shadow price function
figure
pr = funeval(c,basis,s,1);
plot(s,pr ,'color',[0,0,0])
title('Shadow Price Function')
xlabel('pollutant stock')
ylabel('Shadow Price')



%% SIMULATION

% Simulation parameters
nper = 1000;                              % number of periods simulated
nrep = 5000;                           % number of replications

% Initialize simulation
sinit = smin*ones(nrep,1);              % initial wealths
rng('default')

% Simulate model
[ssim,qsim] = dpsimul(model,basis,nper,sinit,[],s,v,q);


ssim1=ssim(:);
qsim1=qsim(:);

% Calculate the 2.5th percentile
lower_bound = prctile(ssim1, 2.5);

% Calculate the 97.5th percentile
upper_bound = prctile(ssim1, 97.5);

% Display the results
disp(['2.5th percentile of pollutant stock: ', num2str(lower_bound)]);
disp(['97.5th percentile of pollutant stock: ', num2str(upper_bound)]);

% Calculate the 2.5th percentile
lower_bound_q = prctile(qsim1, 2.5);

% Calculate the 97.5th percentile
upper_bound_q = prctile(qsim1, 97.5);

% Display the results
disp(['2.5th percentile of decay rate: ', num2str(lower_bound_q)]);
disp(['97.5th percentile of decay rate: ', num2str(upper_bound_q)]);

pisim = (p*(a(1)-a(2)*qsim(:,nper)-a(3)*ssim(:,nper)-0.5*a(4)*qsim(:,nper).^2-0.5*a(5)*ssim(:,nper).^2-a(6)*qsim(:,nper).*ssim(:,nper))...
        -nu(1)-nu(2)*qsim(:,nper)-f*(b(1)-b(2)*qsim(:,nper)+b(3)*z+b(4)*z*qsim(:,nper)))/(1-delta);

% Calculate the 2.5th percentile
lower_bound_pi = prctile(pisim, 2.5);

% Calculate the 97.5th percentile
upper_bound_pi = prctile(pisim, 97.5);

% Display the results
disp(['2.5th percentile of profit: ', num2str(lower_bound_pi)]);
disp(['97.5th percentile of profit: ', num2str(upper_bound_pi)]);


% Plot simulated and expected state path
figure
hold on

plot(0:nper-1,ssim(1,:),':','color',[0,0,0]);
plot(0:nper-1,ssim(2,:),'color',[0.5,0.5,0.5]);
plot(0:nper-1,mean(ssim),'b--o','color',[0,0,0])
xlabel('Period')
ylabel('Plastic Residual in the Soil')
legend('Representative Path 1','Representative Path 2','Expected Path');

% Plot simulated and expected action path
figure
hold on

plot(0:nper-1,qsim(1,:),':','color',[0,0,0]);
plot(0:nper-1,qsim(2,:),'color',[0.5,0.5,0.5]);
plot(0:nper-1,mean(qsim),'b--o','color',[0,0,0])
xlabel('Period')
ylabel('decay rate')
legend('Representative Path 1','Representative Path 2','Expected Path');


% Ergodic moments

pisim = (p*(a(1)-a(2)*qsim(:,nper)-a(3)*ssim(:,nper)-0.5*a(4)*qsim(:,nper).^2-0.5*a(5)*ssim(:,nper).^2-a(6)*qsim(:,nper).*ssim(:,nper))...
        -nu(1)-nu(2)*qsim(:,nper)-f*(b(1)-b(2)*qsim(:,nper)+b(3)*z+b(4)*z*qsim(:,nper)))/(1-delta);
ssim = ssim(:,nper);
qsim = qsim(:,nper);
savg = mean(ssim); 
qavg = mean(qsim);
piavg = mean(pisim);
sstd = std(ssim); 
qstd = std(qsim); 
pistd = std(pisim);
fprintf('Ergodic Moments\n') 
fprintf('                   Deterministic    Ergodic      Ergodic\n') 
fprintf('                    Steady-State      Mean     Std Deviation\n') 
fprintf('Pollutant Stock       %5.3f          %5.3f         %5.3f\n'  ,[sstar savg sstd])
fprintf('Decay Rate            %5.3f          %5.3f         %5.3f\n',[qstar qavg qstd])
fprintf('profit                %5.3f          %5.3f         %5.3f\n\n',[pistar piavg pistd])




%% SAVE FIGURES
printfigures(mfilename)


%% DPSOLVE FUNCTION FILE
%
%    User-supplied function called by dpsolve that returns the bound,
%    reward, and continuous state transition function values and
%    derivatives with respect to the continuous action x at an arbitrary
%    number ns of states and actions according to the format
%      [out1,out2,out3] = func(flag,s,x,i,j,e,<params>)
%    where s is ns.ds continuous states, x is ns.dx continuous actions, i
%    is ns.1 or scalar discrete states, j is ns.1 or scalar discrete
%    actions, and e is ns.de continuous state transition shocks.

function [out1,out2,out3] = func(flag,s,q,~,~,e,a,b,nu,ita,z,f,p)

switch flag
  case 'b'      % bounds
    out1 = zeros(size(s));
    out2 = ones(size(s));
    out3 = [];
  case 'f'      % reward
    out1 = p*(a(1)-a(2)*q-a(3)*s-0.5*a(4)*q.^2-0.5*a(5)*s.^2-a(6)*q.*s)-nu(1)-nu(2)*q-f*(b(1)-b(2)*q+b(3)*z+b(4)*z*q);
    out2 = -a(2)*p-a(4)*q*p-a(6)*s*p-nu(2)+f*b(2)-b(4)*z*f;
    out3 =  -a(4)*p*ones(size(s));
  case 'g'      % transition
   out1 = ita(1)-ita(2)*q+ita(3)*s-ita(4)*z-ita(5)*q*z+e;
    out2 = (-ita(2)-ita(5)*z)*ones(size(s));
    out3 = zeros(size(s));
end