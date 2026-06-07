clear; clc; close all;
%% 参数配置接口 (Unified Interface)
files  = {'SO3_180.mat', 'SE3_180.mat'};
% 修改标签为 SO(3) x R^3
labels = {'$\mathrm{SO(3)} \times \mathrm{R}^3$', '$\mathrm{SE(3)}$'}; 
% 绘图粗细接口
lw_traj   = 2;     % 主轨迹线粗细
lw_axis   = 2.0;   % 渐变坐标轴线粗细
lw_error  = 3;   % 误差曲线粗细
% 比例与尺寸接口
fig_width  = 30;           % 厘米
fig_ratio  = 1 - 0.618;    % 黄金分割余数比例
fig_height = fig_width * fig_ratio;
% 布局对齐接口
plot_aspect = [1.5 1 0.7]; % 强制控制子图框比例
view_angle  = [45, 45];    % 3D 视角
% 仿真参数
Ns = 40; N_plot = Ns + 1; skip = 1; axis_len = 0.15; 

font_size = 14;
%% 画布初始化
fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2, 2, fig_width, fig_height]);

% --- 兼容性布局：2行4列 ---
t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'tight');

% 定义 Case 对应的角度字符串
case_angles = {'-\ 180.01^\circ', '-\ 179.99^\circ'};

for trial = 1:2
    % --- 轨迹图 (第1, 2 列，每图占 1 格) ---
    for f_idx = 1:2
        nexttile; % 默认占 1x1 格
        ax = gca;
        hold on; grid on; box on;
        
        load(files{f_idx});
        res = data_log{trial};
        xu_raw = reshape(res.xu, [24, length(res.xu)/24]);
        xu = xu_raw(:, 1:N_plot); pos = xu(10:12, :);
        
        % 1. 主轨迹
        plot3(pos(1,:), pos(2,:), pos(3,:), 'color', [1,1,1] * 0.5, 'LineWidth', lw_traj);
        
        % 2. 坐标轴
        base_colors = {[1,0,0], [0,1,0], [0,0,1]};
        for k = 1:skip:N_plot
            R = reshape(xu(1:9, k), [3, 3]); p = pos(:, k);
            alpha = 1; 
            for c = 1:3
                c_val = alpha * base_colors{c} + (1-alpha) * [1, 1, 1];
                line([p(1), p(1)+R(1,c)*axis_len], ...
                     [p(2), p(2)+R(2,c)*axis_len], ...
                     [p(3), p(3)+R(3,c)*axis_len], ...
                     'Color', c_val, 'LineWidth', lw_axis);
            end
        end
        
        title([labels{f_idx}, ' $', case_angles{trial}, '$'], 'Interpreter', 'latex', 'FontSize', font_size);
        xlabel('$X$', 'Interpreter', 'latex');
        ylabel('$Y$', 'Interpreter', 'latex');
        zlabel('$Z$', 'Interpreter', 'latex');
        zticklabels({}); 
        
        view(view_angle(1), view_angle(2)); 
        axis tight;
        set(ax, 'PlotBoxAspectRatio', plot_aspect); 
        daspect([1, 1, 1]); 
        xlim([-0.02, 1.02]); ylim([-0.02, 1.02]);
    end
    
    % --- 误差分量图 (跨 2 列，占 1x2 格) ---
    nexttile([1, 2]); 
    ax_err = gca;
    hold on; grid on; box on;
    
    l_handles = [];
    for f_idx = 1:2
        load(files{f_idx});
        res = data_log{trial};
        xu_raw = reshape(res.xu, [24, length(res.xu)/24]);
        ez_log = zeros(1, N_plot);
        for k = 1:N_plot
            Rk = reshape(xu_raw(1:9, k), [3, 3]);
            phi_skew = logm(Rk); 
            ez_log(k) = phi_skew(2,1); 
        end
        
        if f_idx == 1
            h = plot(0:Ns, ez_log, '-', 'LineWidth', lw_error);
        else
            h = plot(0:Ns, ez_log, '--',  'LineWidth', lw_error);
        end
        l_handles(f_idx) = h;
    end
    
    title(['$e_R(z)$ $', case_angles{trial}, '$'], 'Interpreter', 'latex', 'FontSize', font_size);
    xlabel('Step', 'Interpreter', 'latex');
    ylabel('rad', 'Interpreter', 'latex');
    legend(l_handles, labels, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', font_size);
    yline(0, 'k:', 'HandleVisibility', 'off');
    
    % 保持框体高度对齐，但由于跨了两列，宽度会自然增加
    set(ax_err, 'PlotBoxAspectRatio', [3 1 0.7]); % 这里长宽比改大，因为占了两列
end
%% 保存向量图
exportgraphics(fig, 'SO3_SE3_cmpr.pdf', 'ContentType', 'vector');
disp('PDF Exported Successfully.');