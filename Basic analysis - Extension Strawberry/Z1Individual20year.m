%% This code is modeling the individual farmer maximize the profit subject to the pollutant cumulation by
% choosing the mulch decay rate. Given the optimal disposal strategy is to
% not dispose of all the pre-tilled mulches



function Z1Individual20year
%% FORMULATION
 
format long

% Model parameters
a = [18000 1 0.1 0.005 5 0];
b = [0 0 1237.56 -0.5];
nu =[18621.39 10];
%ita 1 was 3.20, but we added 0.05 of residue which is A base level of residue from PE (or BDM for that matter) whether or not the mulch goes to a landfill

ita = [3.25 0.5 0.1 3.2 -0.5];
delta = 0.9;
w = 0.104;
p = 1.01;
z = 0;
             	
 
%% SOLUTION
    
% Model structure
model.func = @func;                             % model functions
model.params = {a,b,nu,ita,z,w,p};              % function parameters
model.discount = delta;                         % discount factor
model.ds = 1;                                   % dimension of continuous state
model.dx = 1;                                   % dimension of continuous action
model.ni = 0;                                   % number of discrete states
model.nj = 0;                                   % number of discrete actions

% Approximation structure
n    = 100;                                     	% number of collocation nodes
smin = 0;                                       	% minimum state
smax = 50;                                          % maximum state
basis = fundefn('spli',n,smin,smax);                % basis functions

%% SOLUTION
digits(40)
% Steady-state
qstar =max(0,((nu(2)+w*(-b(2)+b(4)*z)+a(2)*p)*(1-delta*ita(3))*(1-ita(3))+(a(6)...
    *(1-delta*ita(3))+delta*a(5)*(-ita(2)-ita(5)*z))*p*(ita(1)-ita(4)*z)+...
    a(3)*delta*(-ita(2)-ita(5)*z)*p*(1-ita(3)))/...
    ((-a(4)*(1-ita(3)*delta)+a(6)*delta*(ita(2)+ita(5)*z))*p*(1-ita(3))+(-a(6)*(1-delta*ita(3))...
    +(a(5)*delta*(ita(2)+ita(5)*z))*p*(-ita(2)-ita(5)*z)))); 	% Choice variable
sstar = (ita(1)-ita(2)*qstar-ita(4)*z-ita(5)*z*qstar)/(1-ita(3));           % State Variable
lstar = p*(-a(3)-a(5)*sstar-a(6)*qstar)/(-delta*ita(3)+1);                            % Shadow price for pollutant
pistar = (p*(a(1)-a(2)*qstar-a(3)*sstar-0.5*a(4)*qstar.^2-0.5*a(5)*sstar.^2-a(6)*qstar.*sstar)...
        -nu(1)-nu(2)*qstar-w*(b(1)-b(2)*qstar+b(3)*z+b(4)*z*qstar))/(1-delta); % The present value of life time profit given the optimal choice

fprintf('Steady States\n') 
fprintf('pollutant Stock         %5.2f\n'  ,sstar)
fprintf('decay rate       %5.2f\n'  ,qstar)
fprintf('Shadow Price  %5.2f\n'  ,lstar)
fprintf('lifetime profit  %5.2f\n'  ,pistar)

% check the condition that the optimal z=0. Check value <0
check = -w*(b(3)+b(4)*qstar)+delta*(-ita(4)-ita(5)*qstar)*lstar;
fprintf('check the condition that the optimal z=0. Check value <0 %5.2f\n' , check)
% Check the condition that the FOC=0
check1=-a(2)*p-a(4)*qstar*p-a(6)*sstar*p-nu(2)+w*b(2)-b(4)*z*w+delta*lstar*(-ita(2)-ita(5)*z);
fprintf('Check the condition that the FOC=0 %5.2f\n' , check1)
% Check model derivatives
dpcheck(model,sstar,qstar);
% Solve collocation equation
[c,s,v,q,resid] = dpsolve(model,basis);


%% ANALYSIS

% Plot optimal policy
figure
hold on
plot(s,q,'color',[0,0,0])
xlabel('pollutant Stock')
ylabel('degradation rate')

% ...plot steady-state action
plotbullet(sstar,qstar)
plotvdash(sstar,qstar)
plothdash(sstar,qstar)
plottext(sstar+0.02,[],'$s^*$')
plottext([],qstar,'$\delta^*$')

% Plot value function
figure
hold on
plot(s,v,'color',[0,0,0])
title('Life-time Profit')
xlabel('pollutant Stock')
ylabel('Current Value of life-time profit')

%  ... plot steady-state value
plotvdash(sstar,pistar)
plotbullet(sstar,pistar)
plottext(sstar,[],'$s^*$')

% Plot shadow price function
figure
hold on
p = funeval(c,basis,s,1);
plot(s,p,'color',[0,0,0])
title('Shadow Price Function')
xlabel('pollutant Stock')
ylabel('Shadow Price')

% ...plot steady-state shadow price
plotvdash(sstar,lstar)
plothdash(sstar,lstar)
plotbullet(sstar,lstar)
plottext(sstar+0.02,[],'$s^*$')
plottext([],lstar,'$\lambda^*$')

% Plot residual
figure
hold on
plot(s,resid,'color',[0,0,0])
plothdash([],0)
xlabel('pollutant Stock')
ylabel('Residual')


%% SIMULATION

% Simulation parameters
nper = 1000;                              % number of periods simulated

% Initialize simulation
sinit = smin;                           % agent possesses minimal wealth

% Simulate model
[ssim,qsim] = dpsimul(model,basis,nper,sinit,[],s,v,q);


% Calculate the 2.5th percentile
lower_bound = prctile(ssim, 2.5);

% Calculate the 97.5th percentile
upper_bound = prctile(ssim, 97.5);

% Display the results
disp(['2.5th percentile of pollutant stock: ', num2str(lower_bound)]);
disp(['97.5th percentile of pollutant stock: ', num2str(upper_bound)]);

% Calculate the 2.5th percentile
lower_bound_q = prctile(qsim, 2.5);

% Calculate the 97.5th percentile
upper_bound_q = prctile(qsim, 97.5);

% Display the results
disp(['2.5th percentile of decay rate: ', num2str(lower_bound_q)]);
disp(['97.5th percentile of decay rate: ', num2str(upper_bound_q)]);

% Plot simulated state and policy paths
figure
hold on
plot(0:nper-1,ssim,'color',[0,0,0])
plot(0:nper-1,qsim,'color',[0.6,0.6,0.6])
xlabel('Period')
ylabel('Value for Pollutant Stock or Decay Rate')

% ...plot steady-state
plothdash([],sstar,[0,0,0])
plothdash([],qstar,[0.6,0.6,0.6])
plottext([],sstar,'$s^*$')
plottext([],qstar,'$\delta^*$')
legend('Pollutant Stock','Decay Rate','$s^*$','$\delta^*$')



%% SAVE FIGURES
printfigures(mfilename)


%% DPSOLVE FUNCTION FILE


function [out1,out2,out3] = func(flag,s,q,~,~,~,a,b,nu,ita,z,w,p)

switch flag
  case 'b'      % bounds
    out1 = zeros(size(s));
    out2 = ones(size(s));
    out3 = [];
  case 'f'      % Profit
    out1 = p*(a(1)-a(2)*q-a(3)*s-0.5*a(4)*q.^2-0.5*a(5)*s.^2-a(6)*q.*s)...
        -nu(1)-nu(2)*q-w*(b(1)-b(2)*q+b(3)*z+b(4)*z*q);
    out2 = -a(2)*p-a(4)*q*p-a(6)*s*p-nu(2)+w*b(2)-b(4)*z*w;
    out3 =  -a(4)*p*ones(size(s));
  case 'g'      % Motion of state variable
    out1 = ita(1)-ita(2)*q+ita(3)*s-ita(4)*z-ita(5)*q*z;
    out2 = (-ita(2)-ita(5)*z)*ones(size(s));
    out3 = zeros(size(s));
end