function suspension_geometry_gui()
%SUSPENSION_GEOMETRY_GUI Interactive 3D suspension hardpoint explorer.

addpath(fileparts(mfilename('fullpath')));

%% ---- Parameter definitions --------------------------------------------
paramDefs = [
    pdef('', 'wheelbase', 'Wheelbase', 1200, 2000, 1580)

    pdef('front', 'trackWidth', 'Track width', 1000, 1400, 1240)
    pdef('front', 'tireDiameter', 'Tire diameter', 400, 650, 520)
    pdef('front', 'scrubRadius', 'Scrub radius', -30, 60, 18)
    pdef('front', 'kpiDeg', 'KPI angle (deg)', 0, 15, 8.5)
    pdef('front', 'casterDeg', 'Caster angle (deg)', 0, 12, 6.0)
    pdef('front', 'rollCenterHeight', 'Roll center height', -50, 150, 35)
    pdef('front', 'instantCenterY', 'FV instant center Y', -800, 800, -310)
    pdef('front', 'lowerBallJointHeight', 'Lower ball joint height', 50, 200, 115)
    pdef('front', 'upperBallJointHeight', 'Upper ball joint height', 200, 400, 315)
    pdef('front', 'lowerInboardY', 'Lower arm inboard Y', 200, 400, 310)
    pdef('front', 'upperInboardY', 'Upper arm inboard Y', 180, 380, 285)

    pdef('rear', 'trackWidth', 'Track width', 1000, 1400, 1210)
    pdef('rear', 'tireDiameter', 'Tire diameter', 400, 650, 520)
    pdef('rear', 'scrubRadius', 'Scrub radius', -30, 60, 0)
    pdef('rear', 'rollCenterHeight', 'Roll center height', -50, 150, 55)
    pdef('rear', 'instantCenterY', 'FV instant center Y', -800, 800, -300)
    pdef('rear', 'lowerBallJointHeight', 'Lower ball joint height', 50, 200, 125)
    pdef('rear', 'upperBallJointHeight', 'Upper ball joint height', 200, 400, 325)
    pdef('rear', 'lowerInboardY', 'Lower arm inboard Y', 200, 400, 315)
    pdef('rear', 'upperInboardY', 'Upper arm inboard Y', 180, 380, 295)
];

%% ---- Window & top-level layout -----------------------------------------
fig = uifigure('Name', 'Suspension Geometry Explorer', 'Position', [80 60 1280 800]);
mainGrid = uigridlayout(fig, [1, 2]);
mainGrid.ColumnWidth = {360, '1x'};
mainGrid.RowHeight = {'1x'};

% -- Left: scrollable control panel ---------------------------------------
controlPanel = uipanel(mainGrid, 'Title', 'Package parameters', 'Scrollable', 'on');
controlPanel.Layout.Row = 1;
controlPanel.Layout.Column = 1;

sections = {'', 'front', 'rear'};
sectionTitles = containers.Map({'', 'front', 'rear'}, {'General', 'Front axle', 'Rear axle'});

nRows = numel(paramDefs) + numel(sections); 
ctrlGrid = uigridlayout(controlPanel, [nRows, 3]);
ctrlGrid.ColumnWidth = {140, '1x', 62};
ctrlGrid.RowHeight = repmat({24}, 1, nRows);
ctrlGrid.RowSpacing = 4;

sliderHandles = struct();
fieldHandles = struct();
rowIdx = 0;
for s = 1:numel(sections)
    grp = sections{s};
    rowIdx = rowIdx + 1;
    hdr = uilabel(ctrlGrid, 'Text', sectionTitles(grp), 'FontWeight', 'bold');
    hdr.Layout.Row = rowIdx;
    hdr.Layout.Column = [1 3];

    for k = 1:numel(paramDefs)
        d = paramDefs(k);
        if ~strcmp(d.group, grp)
            continue
        end
        rowIdx = rowIdx + 1;
        key = ctrlKey(d);

        lbl = uilabel(ctrlGrid, 'Text', d.label);
        lbl.Layout.Row = rowIdx;
        lbl.Layout.Column = 1;

        sld = uislider(ctrlGrid, 'Limits', [d.lo, d.hi], 'Value', d.default, ...
            'MajorTicks', [], 'MinorTicks', []);
        sld.Layout.Row = rowIdx;
        sld.Layout.Column = 2;

        ef = uieditfield(ctrlGrid, 'numeric', 'Limits', [d.lo, d.hi], ...
            'Value', d.default, 'ValueDisplayFormat', '%.1f');
        ef.Layout.Row = rowIdx;
        ef.Layout.Column = 3;

        sld.ValueChangingFcn = @(src, evt) onSliderChanging(key, evt.Value);
        sld.ValueChangedFcn  = @(src, evt) onSliderChanged(key, evt.Value);
        ef.ValueChangedFcn   = @(src, evt) onFieldChanged(key, evt.Value);

        sliderHandles.(key) = sld;
        fieldHandles.(key) = ef;
    end
end

% -- Right: 3D view + status/buttons --------------------------------------
rightGrid = uigridlayout(mainGrid, [3, 1]);
rightGrid.Layout.Row = 1;
rightGrid.Layout.Column = 2;
rightGrid.RowHeight = {'1x', 26, 70};

ax = uiaxes(rightGrid);
ax.Layout.Row = 1;
ax.Layout.Column = 1;
xlabel(ax, 'x (forward)');
ylabel(ax, 'y (left)');
zlabel(ax, 'z (up)');
grid(ax, 'on');
axis(ax, 'equal');
view(ax, 35, 22);
title(ax, 'Origin: front center contact patch [0,0,0]');

toolbarGrid = uigridlayout(rightGrid, [1, 3]);
toolbarGrid.Layout.Row = 2;
toolbarGrid.Layout.Column = 1;
toolbarGrid.ColumnWidth = {120, 140, '1x'};
toolbarGrid.Padding = [0 0 0 0];

resetBtn = uibutton(toolbarGrid, 'Text', 'Reset to defaults', 'ButtonPushedFcn', @(s,e) onReset());
resetBtn.Layout.Row = 1; resetBtn.Layout.Column = 1;

exportBtn = uibutton(toolbarGrid, 'Text', 'Export CSV...', 'ButtonPushedFcn', @(s,e) onExport());
exportBtn.Layout.Row = 1; exportBtn.Layout.Column = 2;

statusLabel = uilabel(rightGrid, 'Text', 'Ready.', 'FontColor', [0.15 0.15 0.15]);
statusLabel.Layout.Row = 3;
statusLabel.Layout.Column = 1;
statusLabel.WordWrap = 'on';

%% ---- State ---------------------------------------------------------------
currentPoints = [];
currentReport = [];

updateGeometry();

%% ---- Callbacks -----------------------------------------------------------
    function onSliderChanging(key, val)
        fieldHandles.(key).Value = val;
        updateGeometry();
    end

    function onSliderChanged(key, val)
        fieldHandles.(key).Value = val;
        updateGeometry();
    end

    function onFieldChanged(key, val)
        sliderHandles.(key).Value = clampToLimits(sliderHandles.(key), val);
        updateGeometry();
    end

    function onReset()
        for k = 1:numel(paramDefs)
            d = paramDefs(k);
            key = ctrlKey(d);
            sliderHandles.(key).Value = d.default;
            fieldHandles.(key).Value = d.default;
        end
        updateGeometry();
    end

    function onExport()
        if isempty(currentPoints)
            return
        end
        [file, folder] = uiputfile('suspension_hardpoints.csv', 'Export hardpoints');
        if isequal(file, 0)
            return
        end
        try
            writetable(currentPoints, fullfile(folder, file));
            statusLabel.Text = ['Exported: ' fullfile(folder, file)];
            statusLabel.FontColor = [0.1 0.4 0.1];
        catch ME
            statusLabel.Text = ['Export failed: ' ME.message];
            statusLabel.FontColor = [0.8 0.1 0.1];
        end
    end

    function updateGeometry()
        p = struct();
        for k = 1:numel(paramDefs)
            d = paramDefs(k);
            key = ctrlKey(d);
            val = sliderHandles.(key).Value;
            if isempty(d.group)
                p.(d.field) = val;
            else
                p.(d.group).(d.field) = val;
            end
        end

        try
            [points, geometry, report] = generate_suspension_geometry(p, struct());
            drawGeometry(points, geometry, p);
            allPass = all(report.summary.Pass);
            maxErr = max(abs(report.summary.Error));
            if allPass
                statusLabel.FontColor = [0.1 0.4 0.1];
            else
                statusLabel.FontColor = [0.8 0.4 0.0];
            end
            statusLabel.Text = sprintf('Constraints pass: %d/%d  |  max residual %.2g mm  |  origin = front center contact patch', ...
                sum(report.summary.Pass), numel(report.summary.Pass), maxErr);
            currentPoints = points;
            currentReport = report;
        catch ME
            statusLabel.FontColor = [0.8 0.1 0.1];
            statusLabel.Text = ['Invalid geometry: ' ME.message];
        end
    end

    function drawGeometry(points, geometry, p)
        cla(ax);
        hold(ax, 'on');
        
        % Plot Points
        scatter3(ax, points.X, points.Y, points.Z, 26, [0.1 0.1 0.1], 'filled');
        
        % Plot Point Names
        for i = 1:height(points)
            text(ax, points.X(i) + 5, points.Y(i) + 5, points.Z(i) + 5, points.Name{i}, ...
                'FontSize', 7, 'Color', [0.4 0.4 0.4], 'Interpreter', 'none');
        end

        % Plot Suspensions
        drawCorner(geometry.front.left);
        drawCorner(geometry.front.right);
        drawCorner(geometry.rear.left);
        drawCorner(geometry.rear.right);
        
        % Plot Wheels
        drawWheel(ax, geometry.front.left.wheelCenter, p.front.tireDiameter);
        drawWheel(ax, geometry.front.right.wheelCenter, p.front.tireDiameter);
        drawWheel(ax, geometry.rear.left.wheelCenter, p.rear.tireDiameter);
        drawWheel(ax, geometry.rear.right.wheelCenter, p.rear.tireDiameter);

        % Mark Origin explicitly
        scatter3(ax, 0, 0, 0, 60, [0.85 0.1 0.1], 'filled', 'Marker', 'o');
        hold(ax, 'off');
        grid(ax, 'on');
        axis(ax, 'equal');
    end

    function drawCorner(c)
        line3(c.lowerBallJoint, c.lowerInboardForward, [0.1 0.3 0.9]);
        line3(c.lowerBallJoint, c.lowerInboardAft, [0.1 0.3 0.9]);
        line3(c.upperBallJoint, c.upperInboardForward, [0.1 0.6 0.4]);
        line3(c.upperBallJoint, c.upperInboardAft, [0.1 0.6 0.4]);
        line3(c.lowerBallJoint, c.upperBallJoint, [0.1 0.1 0.1]);
        if isfield(c, 'directDamperOutboard')
            line3(c.directDamperOutboard, c.directDamperInboard, [0.8 0.1 0.1]);
        end
        if isfield(c, 'pushrodOutboard')
            line3(c.pushrodOutboard, c.rockerPushrodEnd, [0.8 0.1 0.1]);
            line3(c.rockerPivot, c.rockerPushrodEnd, [0.6 0.2 0.7]);
            line3(c.rockerPivot, c.rockerDamperEnd, [0.6 0.2 0.7]);
            line3(c.rockerPivot, c.rockerArbEnd, [0.6 0.2 0.7]);
            line3(c.rockerDamperEnd, c.rearDamperInboard, [0.8 0.1 0.1]);
            line3(c.rockerArbEnd, c.arbTubeArmEnd, [0.2 0.2 0.2]);
        end
    end

    function line3(a, b, color)
        plot3(ax, [a(1) b(1)], [a(2) b(2)], [a(3) b(3)], 'Color', color, 'LineWidth', 1.6);
    end

    function drawWheel(ax, center, diameter)
        theta = linspace(0, 2*pi, 60);
        r = diameter / 2;
        x = center(1) + r * cos(theta);
        y = repmat(center(2), 1, 60);
        z = center(3) + r * sin(theta);
        plot3(ax, x, y, z, '-', 'Color', [0.2 0.2 0.2 0.5], 'LineWidth', 2);
    end

end

function v = clampToLimits(sliderHandle, v)
lims = sliderHandle.Limits;
v = min(max(v, lims(1)), lims(2));
end

function key = ctrlKey(d)
if isempty(d.group)
    key = d.field;
else
    key = [d.group '_' d.field];
end
end

function d = pdef(group, field, label, lo, hi, default)
d = struct('group', group, 'field', field, 'label', label, ...
    'lo', lo, 'hi', hi, 'default', default);
end