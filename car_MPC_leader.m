function [out_vec] = car_MPC_leader(x0,w0,dims,t)
    
    % Minimize the YALMIP overhead by defining persistent optimizers

    if exist('Controller', 'var') < 1
        persistent Controller
    end
    
    n = dims(1);
    m = dims(2);

    % At simulation startup, initialize the optimizers and then merely use
    % them during subsequent iterations.
    
    if t == 0
        
        % Bring up the car's closed-loop system (with its NRF subcontroller)
        % from the main workspace.

        A_area   = evalin('base','A_area');
        B_area   = evalin('base','B_area');
        B_area_p = evalin('base','B_area_p');
        h        = evalin('base','h');
    
        n_c      = length(A_area{1}) - n;
        [~, m_w] = size(B_area_p);
        [~, m_u] = size(B_area{1});

        % Bring up the constraints in H-representation, along with the cost
        % weighting matrices.
    
        H_U    = evalin('base','H_U');
        w_U    = evalin('base','w_U');
        H_J    = evalin('base','H_J');
        w_J    = evalin('base','w_J');
        Q_cost = evalin('base','Q_cost');
        R_cost = evalin('base','R_cost');

        % Clean up the environment and set up auxiliary variables for a
        % one-step prediction horizon (we are only interested in recursive
        % constraint satisfaction).
        
        yalmip('clear')
        
        u   = sdpvar(m_u,1);
        x   = sdpvar(n+n_c,1);
        x_n = sdpvar(n+n_c,1);
        w   = sdpvar(m_w,1);

        % Form the quadratic cost function and set up the dynamical and
        % set-based constraints.
        
        objective = Q_cost{1} * ((x_n(1) + h * x_n(2))^2) +...
                    u' * R_cost{1} * u;
        constraints = x_n == A_area{1} * x + B_area{1} * u + B_area_p * w;
        constraints = [constraints, H_U{1}*u   <= w_U{1}];
        constraints = [constraints, H_J{1}*x_n <= w_J{1}];

        % Form the optimizer-based controller.

        ops = sdpsettings('verbose',2);
        Controller{1} = optimizer(constraints,objective,ops,{x,w},u);
       
    end

    % Use the controller to compute the command signals.
    
    [uout,problem] = Controller{1}(x0,w0);
    if problem > 0 % Set second layer command signals to zero and trigger
                   % the infeasibility flag (since recursive feasibility is
                   % ensured by the employed sets, this branch is associated
                   % with errors emitted by the solver).
       out_vec(1:m) = zeros(m,1);
       out_vec(m+1:m+n-1) = zeros(n-1,1);
       out_vec(m+n) = 1;
    else % No problem detected
       out_vec(1:m) = uout(1:m);
       out_vec(m+1:m+n-1) = uout(m+1:m+n-1);
       out_vec(m+n) = 0;
    end

end