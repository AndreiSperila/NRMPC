function [out_vec] = car_MPC_follower(x0,w0,dims,t)
    
    % Minimize the YALMIP overhead by defining persistent optimizers
    
    if exist('Controller', 'var') < 1
        persistent Controller
    end
    
    car_id = dims(1);
    n      = dims(2);
    m      = dims(3);
    
    % At simulation startup, initialize the optimizers and then merely use
    % them during subsequent iterations.

    if t == 0

        % Bring up the car's closed-loop system (with its NRF subcontroller)
        % from the main workspace.
    
        A_area   = evalin('base','A_area');
        B_area   = evalin('base','B_area');
        A_area_p = evalin('base','A_area_p');
        h        = evalin('base','h');
        
        n_c      = length(A_area{car_id}) - n;
        [~, m_w] = size(A_area_p{car_id});
        [~, m_u] = size(B_area{car_id});

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

        objective = Q_cost{car_id} * ((x_n(1) + h * x_n(2))^2) +...
                    u' * R_cost{car_id} * u;
        constraints = x_n == A_area{car_id} * x + B_area{car_id} * u +...
                             A_area_p{car_id} * w;
        constraints = [constraints, H_U{car_id}*u   <= w_U{car_id}];
        constraints = [constraints, H_J{car_id}*x_n <= w_J{car_id}];
        
        % Form the optimizer-based controller.

        ops = sdpsettings('verbose',2);
        Controller{car_id} = optimizer(constraints,objective,ops,{x,w},u);
        
    end
    
    % Use the controller to compute the command signals.

    [uout,problem] = Controller{car_id}(x0,w0);
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