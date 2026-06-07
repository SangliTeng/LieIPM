clear; clc; close all;
%% 参数配置接口
files  = {'SO3_180.mat', 'SE3_180.mat'};
files_many = {'SO3_many.mat', 'SE3_many.mat'}; % 新增数据集接口
labels = {'$\mathrm{SO(3)} \times \mathrm{R}^3$', '$\mathrm{SE(3)}$'};
% 绘图粗细与尺寸
lw_loss    = 2.0;    % KKT Loss 曲线粗细
lw_many    = 1.0;    % many 数据集曲线稍细
fig_width  = 20;     % 厘米 (两列布局)
fig_ratio  = 0.5;    % 宽高比
fig_height = fig_width * fig_ratio;
%% 画布初始化
fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [5, 5, fig_width, fig_height]);
t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'tight');
% 定义 Case 角度用于标注
case_angles = {'180.01^\circ', '179.99^\circ'};
colors = {'r', 'b'}; 
for f_idx = 1:2
    ax = nexttile;
    hold on; grid on; box on;
    
    % --- 1. 绘制 many 数据集 (背景) ---
    if exist(files_many{f_idx}, 'file')
        m_data = load(files_many{f_idx});
        % 假设 many 文件里也是 data_log 结构
        for m_idx = 1:length(m_data.data_log)
            m_res = m_data.data_log{m_idx};
            m_kkt = m_res.overall_kkt_violation_log;
            % 默认颜色，虚线，不进入图例
            semilogy(0:length(m_kkt)-1, m_kkt, '--', 'LineWidth', lw_many, 'HandleVisibility', 'off');
        end
    end

    % --- 2. 绘制原本的 Trial 数据 ---
    load(files{f_idx});
    l_handles = [];
    for trial = 1:2
        res = data_log{trial};
        
        % 获取 KKT Loss 数据
        kkt_data = res.overall_kkt_violation_log;
        iter_axis = 0:length(kkt_data)-1;
        
        % --- 线型逻辑：第一条实线，第二条点划线 -- ---
        if trial == 1
            line_style = '-';
        else
            line_style = '--';
        end
        
        h = semilogy(iter_axis, kkt_data, line_style, 'Color', colors{trial}, 'LineWidth', lw_loss);
        l_handles(trial) = h;
    end
    
    % --- 在 1e-12 处加入黑色虚线 ---
    yline(1e-12, 'k--', 'LineWidth', 1.0, 'HandleVisibility', 'off');
    
    % 强制 Y 轴对数刻度与网格显示
    set(ax, 'YScale', 'log'); 
    grid(ax, 'on');
    
    % 装饰
    title(labels{f_idx}, 'Interpreter', 'latex', 'FontSize', 13);
    xlabel('Iteration', 'Interpreter', 'latex');
    ylabel('Overall KKT Loss', 'Interpreter', 'latex');
    
    % 图例
    legend(l_handles, {['$\alpha = ', case_angles{1}, '$'], ...
                       ['$\alpha = ', case_angles{2}, '$']}, ...
           'Interpreter', 'latex', 'Location', 'northeast');
    
    set(ax, 'TickLabelInterpreter', 'latex');
    
    % 移除 PlotBoxAspectRatio 限制，允许高度填满
    set(ax, 'PlotBoxAspectRatioMode', 'auto'); 
    if f_idx == 2
        xlim([0, 50])
    else
        xlim([0, 25])
    end
    ylim([1e-16, 1e5])
end
%% 保存向量图
exportgraphics(fig, 'KKT_Loss_Comparison.pdf', 'ContentType', 'vector');
disp('PDF Exported Successfully: KKT_Loss_Comparison.pdf');