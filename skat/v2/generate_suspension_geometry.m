function [points, geometry, report] = generate_suspension_geometry(p, manual)
%GENERATE_SUSPENSION_GEOMETRY Build suspension hardpoints from package targets.
%
%   [points, geometry, report] = generate_suspension_geometry(p, manual)
%
%   Coordinate system:
%     x forward, y left, z up. Origin is the center of the front axle 
%     on the ground (i.e. Front center contact patch = [0, 0, 0]).
%     Default units are millimeters and degrees.
%
%   Primary input fields, all optional:
%     p.wheelbase
%     p.front.trackWidth, p.rear.trackWidth
%     p.front.tireDiameter, p.rear.tireDiameter
%     p.front.scrubRadius, p.rear.scrubRadius
%     p.front.kpiDeg, p.rear.kpiDeg
%     p.front.casterDeg, p.rear.casterDeg
%     p.front.rollCenterHeight, p.rear.rollCenterHeight
%
%   Useful package fields:
%     p.front.lowerBallJointHeight, p.front.upperBallJointHeight
%     p.front.lowerInboardY, p.front.upperInboardY
%     p.front.instantCenterY, p.rear.instantCenterY
%       Left-side front-view instant-center y coordinate. Negative values put
%       the left IC on the right side of the car. The right side is mirrored.
%
%   Manual point fields, all optional:
%     manual.frontDirectInboardLeft,  manual.frontDirectInboardRight
%     manual.rearRockerPivotLeft,     manual.rearRockerPivotRight
%     manual.rearRockerPlaneGuideLeft, manual.rearRockerPlaneGuideRight
%     manual.rearDamperInboardLeft,   manual.rearDamperInboardRight
%     manual.rearArbTubeArmEndLeft,   manual.rearArbTubeArmEndRight
%
%   Rear rocker layout fields:
%     p.rear.rockerPushrodAngleDeg, p.rear.rockerDamperAngleDeg,
%     p.rear.rockerArbAngleDeg
%       Angles are in the rocker plane, measured from the pivot-to-pushrod
%       outboard direction toward the generated in-plane e2 direction.

if nargin < 1 || isempty(p)
    p = struct();
end
if nargin < 2 || isempty(manual)
    manual = struct();
end

userManual = manual;
p = merge_defaults(default_parameters(), p);
p = finalize_parameters(p);
validate_parameters(p);

manual = merge_defaults(default_manual_points(p), manual);
manual = mirror_missing_manual_pairs(manual, userManual);

[rows, geometry, report] = build_geometry(p, manual);

% Re-origin everything to the front center contact patch (ground center).
originOffset = [p.front.axleX, 0, 0];
rows = translate_rows(rows, originOffset);
geometry.front = translate_axle_geometry(geometry.front, originOffset);
geometry.rear = translate_axle_geometry(geometry.rear, originOffset);
report.origin = 'front center contact patch (Front_Axle_Ground_Center = [0,0,0])';

points = rows_to_table(rows);

if ~isempty(p.outputCsv)
    writetable(points, p.outputCsv);
end

if nargout == 0
    disp(points);
    disp(report.summary);
    clear points geometry report
end
end

function p = default_parameters()
p.units = 'mm';
p.coordinateSystem = 'x forward, y left, z up; origin at front center contact patch';
p.outputCsv = '';
p.geometryTolerance = 1.0e-6;
p.projectManualRockerPointsToPlane = true;
p.wheelbase = 1600;

p.front = default_axle_parameters();
p.front.axleX = 0;
p.front.trackWidth = 1220;
p.front.tireDiameter = 520;
p.front.scrubRadius = 20;
p.front.kpiDeg = 8;
p.front.casterDeg = 6;
p.front.rollCenterHeight = 35;
p.front.lowerBallJointHeight = 115;
p.front.upperBallJointHeight = 315;
p.front.lowerInboardY = 310;
p.front.upperInboardY = 285;
p.front.lowerArmHalfForeAft = 155;
p.front.upperArmHalfForeAft = 125;
p.front.directMountFractionFromLBJ = 0.0; % In line with balljoint

p.rear = default_axle_parameters();
p.rear.axleX = [];
p.rear.trackWidth = 1180;
p.rear.tireDiameter = 520;
p.rear.scrubRadius = 0;
p.rear.kpiDeg = 0;
p.rear.casterDeg = 0;
p.rear.rollCenterHeight = 55;
p.rear.lowerBallJointHeight = 125;
p.rear.upperBallJointHeight = 325;
p.rear.lowerInboardY = 315;
p.rear.upperInboardY = 295;
p.rear.lowerArmHalfForeAft = 165;
p.rear.upperArmHalfForeAft = 130;
p.rear.pushrodMountFractionFromUBJ = 0.0; % In line with balljoint
p.rear.rockerPushrodArmLength = 75;
p.rear.rockerDamperArmLength = 70;
p.rear.rockerArbArmLength = 60;
p.rear.rockerPushrodAngleDeg = 28;
p.rear.rockerDamperAngleDeg = 112;
p.rear.rockerArbAngleDeg = -60; % Bottom and outboard of pivot
end

function a = default_axle_parameters()
a.axleX = [];
a.trackWidth = [];
a.tireDiameter = [];
a.scrubRadius = 0;
a.kpiDeg = 0;
a.casterDeg = 0;
a.steeringAxisGroundXOffset = 0;
a.rollCenterHeight = [];
a.instantCenterY = [];
a.lowerBallJointHeight = [];
a.upperBallJointHeight = [];
a.lowerInboardY = [];
a.upperInboardY = [];
a.lowerInboardXOffset = 0;
a.upperInboardXOffset = 0;
a.lowerArmHalfForeAft = [];
a.upperArmHalfForeAft = [];
a.lowerInboardZDeltaForeMinusAft = 0;
a.upperInboardZDeltaForeMinusAft = 0;
a.directMountFractionFromLBJ = 0.0;
a.pushrodMountFractionFromUBJ = 0.0;
a.rockerPushrodArmLength = 75;
a.rockerDamperArmLength = 70;
a.rockerArbArmLength = 60;
a.rockerPushrodAngleDeg = 28;
a.rockerDamperAngleDeg = 112;
a.rockerArbAngleDeg = -60;
end

function p = finalize_parameters(p)
if isempty(p.front.axleX)
    p.front.axleX = 0;
end
if isempty(p.rear.axleX)
    p.rear.axleX = -p.wheelbase;
end
p.front = finalize_axle_parameters(p.front);
p.rear = finalize_axle_parameters(p.rear);
end

function a = finalize_axle_parameters(a)
if isempty(a.instantCenterY)
    a.instantCenterY = -0.25 * a.trackWidth;
end
end

function manual = default_manual_points(p)
manual.frontDirectInboardLeft = [p.front.axleX + 45, 260, 610];
manual.frontDirectInboardRight = mirror_y(manual.frontDirectInboardLeft);

manual.rearRockerPivotLeft = [p.rear.axleX + 85, 280, 560];
manual.rearRockerPivotRight = mirror_y(manual.rearRockerPivotLeft);
manual.rearRockerPlaneGuideLeft = [];
manual.rearRockerPlaneGuideRight = [];

manual.rearDamperInboardLeft = [p.rear.axleX - 210, 245, 545];
manual.rearDamperInboardRight = mirror_y(manual.rearDamperInboardLeft);

manual.rearArbTubeArmEndLeft = [p.rear.axleX + 45, 120, 420];
manual.rearArbTubeArmEndRight = mirror_y(manual.rearArbTubeArmEndLeft);
end

function [rows, geometry, report] = build_geometry(p, manual)
rows = cell(0, 8);
geometry = struct();
report = struct();
report.units = p.units;
report.coordinateSystem = p.coordinateSystem;
report.tolerance = p.geometryTolerance;

[rows, geometry.front, report.front] = build_axle(rows, p, manual, 'front');
[rows, geometry.rear, report.rear] = build_axle(rows, p, manual, 'rear');

report.summary = summarize_report(report, p.geometryTolerance);
end

function [rows, axleGeometry, axleReport] = build_axle(rows, p, manual, axleName)
a = p.(axleName);
axleTitle = title_case(axleName);
axleGeometry = struct();
axleReport = struct();

rows = append_point(rows, [axleTitle '_RollCenter'], axleName, 'Center', ...
    'Reference', [a.axleX, 0, a.rollCenterHeight], ...
    'Target roll-center point.');

for side = [1, -1]
    sideField = side_field(side);
    [rows, corner, cornerReport] = build_corner_base(rows, a, axleName, side);
    if strcmp(axleName, 'front')
        [rows, corner, cornerReport] = add_front_actuation(rows, a, manual, side, corner, cornerReport);
    else
        [rows, corner, cornerReport] = add_rear_actuation(rows, p, a, manual, side, corner, cornerReport);
    end
    axleGeometry.(sideField) = corner;
    axleReport.(sideField) = cornerReport;
end
end

function [rows, c, r] = build_corner_base(rows, a, axleName, side)
axleTitle = title_case(axleName);
sideName = side_token(side);
prefix = [axleTitle '_' sideName '_'];

contactPatch = [a.axleX, side * a.trackWidth / 2, 0];
wheelCenter = [a.axleX, side * a.trackWidth / 2, a.tireDiameter / 2];
lowerBJ = steering_axis_point(a, side, a.lowerBallJointHeight);
upperBJ = steering_axis_point(a, side, a.upperBallJointHeight);

icY = side * a.instantCenterY;
cpY = contactPatch(2);
if abs(icY - cpY) < 1.0e-9
    error('%s instantCenterY cannot place the IC at the contact patch y coordinate.', axleTitle);
end
icZ = a.rollCenterHeight * (1 - icY / cpY);
instantCenter = [a.axleX, icY, icZ];

lowerInY = side * a.lowerInboardY;
upperInY = side * a.upperInboardY;
lowerInZ = line_z_at_y([lowerBJ(2), lowerBJ(3)], [icY, icZ], lowerInY);
upperInZ = line_z_at_y([upperBJ(2), upperBJ(3)], [icY, icZ], upperInY);

lowerCenter = [a.axleX + a.lowerInboardXOffset, lowerInY, lowerInZ];
upperCenter = [a.axleX + a.upperInboardXOffset, upperInY, upperInZ];

lowerFore = lowerCenter + [a.lowerArmHalfForeAft, 0, 0.5 * a.lowerInboardZDeltaForeMinusAft];
lowerAft = lowerCenter + [-a.lowerArmHalfForeAft, 0, -0.5 * a.lowerInboardZDeltaForeMinusAft];
upperFore = upperCenter + [a.upperArmHalfForeAft, 0, 0.5 * a.upperInboardZDeltaForeMinusAft];
upperAft = upperCenter + [-a.upperArmHalfForeAft, 0, -0.5 * a.upperInboardZDeltaForeMinusAft];

rows = append_point(rows, [prefix 'ContactPatch'], axleName, sideName, 'Wheel', contactPatch, '');
rows = append_point(rows, [prefix 'WheelCenter'], axleName, sideName, 'Wheel', wheelCenter, '');
rows = append_point(rows, [prefix 'LowerBallJoint'], axleName, sideName, 'Upright', lowerBJ, '');
rows = append_point(rows, [prefix 'UpperBallJoint'], axleName, sideName, 'Upright', upperBJ, '');
rows = append_point(rows, [prefix 'LowerArmInboardForward'], axleName, sideName, 'ControlArm', lowerFore, '');
rows = append_point(rows, [prefix 'LowerArmInboardAft'], axleName, sideName, 'ControlArm', lowerAft, '');
rows = append_point(rows, [prefix 'UpperArmInboardForward'], axleName, sideName, 'ControlArm', upperFore, '');
rows = append_point(rows, [prefix 'UpperArmInboardAft'], axleName, sideName, 'ControlArm', upperAft, '');
rows = append_point(rows, [prefix 'FrontViewInstantCenter'], axleName, sideName, 'Reference', instantCenter, ...
    'Construction point; not a physical pickup.');

c = struct();
c.contactPatch = contactPatch;
c.wheelCenter = wheelCenter;
c.lowerBallJoint = lowerBJ;
c.upperBallJoint = upperBJ;
c.lowerInboardCenter = lowerCenter;
c.upperInboardCenter = upperCenter;
c.lowerInboardForward = lowerFore;
c.lowerInboardAft = lowerAft;
c.upperInboardForward = upperFore;
c.upperInboardAft = upperAft;
c.instantCenter = instantCenter;

r = struct();
r.rollCenterHeightSolved = line_z_at_y([contactPatch(2), contactPatch(3)], [icY, icZ], 0);
r.rollCenterHeightError = r.rollCenterHeightSolved - a.rollCenterHeight;
r.kingpinInclinationDeg = a.kpiDeg;
r.casterDeg = a.casterDeg;
end

function [rows, c, r] = add_front_actuation(rows, a, manual, side, c, r)
axleName = 'front';
sideName = side_token(side);
prefix = ['Front_' sideName '_'];

outboard = c.lowerBallJoint + a.directMountFractionFromLBJ * ...
    (c.lowerInboardCenter - c.lowerBallJoint);
defaultInboard = [a.axleX + 45, side * 260, 610];
inboard = get_manual_point(manual, 'frontDirectInboard', side, defaultInboard);

rows = append_point(rows, [prefix 'DirectDamperOutboard'], axleName, sideName, ...
    'FrontDamper', outboard, 'On lower arm centerline near lower ball joint.');
rows = append_point(rows, [prefix 'DirectDamperInboard'], axleName, sideName, ...
    'FrontDamper', inboard, 'Manual chassis-side direct damper point.');

c.directDamperOutboard = outboard;
c.directDamperInboard = inboard;
c.directDamperLength = norm(inboard - outboard);

r.directOutboardLineError = point_line_distance(outboard, c.lowerBallJoint, c.lowerInboardCenter);
r.directDamperLength = c.directDamperLength;
end

function [rows, c, r] = add_rear_actuation(rows, p, a, manual, side, c, r)
axleName = 'rear';
sideName = side_token(side);
prefix = ['Rear_' sideName '_'];

pushrodOutboard = c.upperBallJoint + a.pushrodMountFractionFromUBJ * ...
    (c.upperInboardCenter - c.upperBallJoint);

defaultPivot = [a.axleX + 85, side * 280, 560];
rockerPivot = get_manual_point(manual, 'rearRockerPivot', side, defaultPivot);
planeGuide = get_manual_point(manual, 'rearRockerPlaneGuide', side, rockerPivot + [0, 0, 100]);
if norm(planeGuide - rockerPivot) < 1.0e-9
    planeGuide = rockerPivot + [0, 0, 100];
end

[planeNormal, e1, e2] = rocker_plane_basis(rockerPivot, pushrodOutboard, planeGuide);
rockerPushrodEnd = rockerPivot + a.rockerPushrodArmLength * ...
    (cosd(a.rockerPushrodAngleDeg) * e1 + sind(a.rockerPushrodAngleDeg) * e2);
rockerDamperEnd = rockerPivot + a.rockerDamperArmLength * ...
    (cosd(a.rockerDamperAngleDeg) * e1 + sind(a.rockerDamperAngleDeg) * e2);
rockerArbEnd = rockerPivot + a.rockerArbArmLength * ...
    (cosd(a.rockerArbAngleDeg) * e1 + sind(a.rockerArbAngleDeg) * e2);

if has_manual_point(manual, ['rearRockerPushrodEnd' sideName])
    rockerPushrodEnd = manual_projected_point(manual, ['rearRockerPushrodEnd' sideName], ...
        rockerPivot, planeNormal, p.projectManualRockerPointsToPlane);
end
if has_manual_point(manual, ['rearRockerDamperEnd' sideName])
    rockerDamperEnd = manual_projected_point(manual, ['rearRockerDamperEnd' sideName], ...
        rockerPivot, planeNormal, p.projectManualRockerPointsToPlane);
end
if has_manual_point(manual, ['rearRockerArbEnd' sideName])
    rockerArbEnd = manual_projected_point(manual, ['rearRockerArbEnd' sideName], ...
        rockerPivot, planeNormal, p.projectManualRockerPointsToPlane);
end

defaultDamperInboard = [a.axleX - 210, side * 245, 545];
damperInboard = get_manual_point(manual, 'rearDamperInboard', side, defaultDamperInboard);
defaultArbTube = [a.axleX + 45, side * 120, 420];
arbTubeArmEnd = get_manual_point(manual, 'rearArbTubeArmEnd', side, defaultArbTube);

rows = append_point(rows, [prefix 'PushrodOutboard'], axleName, sideName, ...
    'RearPushrod', pushrodOutboard, 'On upper arm centerline near upper ball joint.');
rows = append_point(rows, [prefix 'RockerPivot'], axleName, sideName, ...
    'RearRocker', rockerPivot, 'Manual chassis-side rocker pivot.');
rows = append_point(rows, [prefix 'RockerPushrodEnd'], axleName, sideName, ...
    'RearRocker', rockerPushrodEnd, 'Planar with rocker pivot and pushrod outboard.');
rows = append_point(rows, [prefix 'RockerDamperEnd'], axleName, sideName, ...
    'RearRocker', rockerDamperEnd, 'Planar rocker damper pickup.');
rows = append_point(rows, [prefix 'DamperInboard'], axleName, sideName, ...
    'RearDamper', damperInboard, 'Manual chassis-side rear damper point.');
rows = append_point(rows, [prefix 'RockerARBEnd'], axleName, sideName, ...
    'RearARB', rockerArbEnd, 'Planar rocker ARB pickup.');
rows = append_point(rows, [prefix 'ARBTubeArmEnd'], axleName, sideName, ...
    'RearARB', arbTubeArmEnd, 'Manual ARB tube arm/link point.');

c.pushrodOutboard = pushrodOutboard;
c.rockerPivot = rockerPivot;
c.rockerPlaneNormal = planeNormal;
c.rockerPlaneBasis1 = e1;
c.rockerPlaneBasis2 = e2;
c.rockerPushrodEnd = rockerPushrodEnd;
c.rockerDamperEnd = rockerDamperEnd;
c.rearDamperInboard = damperInboard;
c.rockerArbEnd = rockerArbEnd;
c.arbTubeArmEnd = arbTubeArmEnd;
c.pushrodLength = norm(rockerPushrodEnd - pushrodOutboard);
c.rearDamperLength = norm(damperInboard - rockerDamperEnd);

planarPoints = [
    rockerPivot;
    pushrodOutboard;
    rockerPushrodEnd;
    rockerDamperEnd;
    rockerArbEnd
];

r.pushrodOutboardLineError = point_line_distance(pushrodOutboard, c.upperBallJoint, c.upperInboardCenter);
r.rockerPlanarityMaxError = max(abs(point_plane_distances(planarPoints, rockerPivot, planeNormal)));
r.pushrodRockerMomentArm = point_line_distance(rockerPivot, pushrodOutboard, rockerPushrodEnd);
r.pushrodLength = c.pushrodLength;
r.rearDamperLength = c.rearDamperLength;
end

function rows = translate_rows(rows, offset)
for i = 1:size(rows, 1)
    rows{i, 5} = rows{i, 5} - offset(1);
    rows{i, 6} = rows{i, 6} - offset(2);
    rows{i, 7} = rows{i, 7} - offset(3);
end
end

function axleGeometry = translate_axle_geometry(axleGeometry, offset)
axleGeometry.left = translate_corner(axleGeometry.left, offset);
axleGeometry.right = translate_corner(axleGeometry.right, offset);
end

function c = translate_corner(c, offset)
pointFields = {
    'contactPatch', 'wheelCenter', 'lowerBallJoint', 'upperBallJoint', ...
    'lowerInboardCenter', 'upperInboardCenter', ...
    'lowerInboardForward', 'lowerInboardAft', ...
    'upperInboardForward', 'upperInboardAft', 'instantCenter', ...
    'directDamperOutboard', 'directDamperInboard', ...
    'pushrodOutboard', 'rockerPivot', 'rockerPushrodEnd', 'rockerDamperEnd', ...
    'rearDamperInboard', 'rockerArbEnd', 'arbTubeArmEnd'
};
for k = 1:numel(pointFields)
    fn = pointFields{k};
    if isfield(c, fn)
        c.(fn) = c.(fn) - offset;
    end
end
end

function pt = steering_axis_point(a, side, z)
axisGroundX = a.axleX + a.steeringAxisGroundXOffset;
axisGroundY = side * (a.trackWidth / 2 - a.scrubRadius);
x = axisGroundX - tand(a.casterDeg) * z;
y = axisGroundY - side * tand(a.kpiDeg) * z;
pt = [x, y, z];
end

function [normal, e1, e2] = rocker_plane_basis(pivot, pushrodOutboard, guide)
e1 = unit_vector(pushrodOutboard - pivot);
guideVector = guide - pivot;
if norm(cross(e1, guideVector)) < 1.0e-9
    guideVector = [0, 0, 1];
end
if norm(cross(e1, guideVector)) < 1.0e-9
    guideVector = [1, 0, 0];
end
normal = unit_vector(cross(e1, guideVector));
e2 = unit_vector(cross(normal, e1));
if e2(3) < 0
    e2 = -e2;
    normal = -normal;
end
end

function points = rows_to_table(rows)
Name = rows(:, 1);
Axle = rows(:, 2);
Side = rows(:, 3);
System = rows(:, 4);
X = cell2mat(rows(:, 5));
Y = cell2mat(rows(:, 6));
Z = cell2mat(rows(:, 7));
Note = rows(:, 8);
points = table(Name, Axle, Side, System, X, Y, Z, Note);
end

function rows = append_point(rows, name, axleName, sideName, systemName, pt, note)
pt = as_point(pt, name);
rows(end + 1, :) = {name, axleName, sideName, systemName, pt(1), pt(2), pt(3), note};
end

function summary = summarize_report(report, tolerance)
Constraint = {
    'Front left direct pickup on lower arm line';
    'Front right direct pickup on lower arm line';
    'Rear left pushrod pickup on upper arm line';
    'Rear right pushrod pickup on upper arm line';
    'Rear left rocker and pushrod planarity';
    'Rear right rocker and pushrod planarity';
    'Front left roll center height';
    'Front right roll center height';
    'Rear left roll center height';
    'Rear right roll center height'
};
Error = [
    report.front.left.directOutboardLineError;
    report.front.right.directOutboardLineError;
    report.rear.left.pushrodOutboardLineError;
    report.rear.right.pushrodOutboardLineError;
    report.rear.left.rockerPlanarityMaxError;
    report.rear.right.rockerPlanarityMaxError;
    abs(report.front.left.rollCenterHeightError);
    abs(report.front.right.rollCenterHeightError);
    abs(report.rear.left.rollCenterHeightError);
    abs(report.rear.right.rollCenterHeightError)
];
Tolerance = tolerance * ones(size(Error));
Pass = Error <= Tolerance;
summary = table(Constraint, Error, Tolerance, Pass);
end

function z = line_z_at_y(yz1, yz2, y)
den = yz2(1) - yz1(1);
if abs(den) < 1.0e-9
    error('Cannot solve line z at y because the two front-view y coordinates are identical.');
end
z = yz1(2) + (yz2(2) - yz1(2)) * (y - yz1(1)) / den;
end

function d = point_line_distance(point, lineA, lineB)
ab = lineB - lineA;
if norm(ab) < 1.0e-12
    d = norm(point - lineA);
else
    d = norm(cross(point - lineA, ab)) / norm(ab);
end
end

function distances = point_plane_distances(points, planePoint, planeNormal)
normal = unit_vector(planeNormal);
distances = (points - planePoint) * normal(:);
end

function pt = project_point_to_plane(pt, planePoint, planeNormal)
normal = unit_vector(planeNormal);
pt = pt - dot(pt - planePoint, normal) * normal;
end

function pt = manual_projected_point(manual, fieldName, planePoint, planeNormal, doProject)
pt = as_point(manual.(fieldName), fieldName);
if doProject
    pt = project_point_to_plane(pt, planePoint, planeNormal);
end
end

function tf = has_manual_point(manual, fieldName)
tf = isfield(manual, fieldName) && ~isempty(manual.(fieldName));
end

function pt = get_manual_point(manual, baseName, side, defaultPoint)
fieldName = [baseName side_token(side)];
if has_manual_point(manual, fieldName)
    pt = as_point(manual.(fieldName), fieldName);
else
    pt = as_point(defaultPoint, [fieldName ' default']);
end
end

function manual = mirror_missing_manual_pairs(manual, userManual)
baseNames = {
    'frontDirectInboard';
    'rearRockerPivot';
    'rearRockerPlaneGuide';
    'rearDamperInboard';
    'rearArbTubeArmEnd';
    'rearRockerPushrodEnd';
    'rearRockerDamperEnd';
    'rearRockerArbEnd'
};

for k = 1:numel(baseNames)
    leftName = [baseNames{k} 'Left'];
    rightName = [baseNames{k} 'Right'];
    leftGiven = isfield(userManual, leftName) && ~isempty(userManual.(leftName));
    rightGiven = isfield(userManual, rightName) && ~isempty(userManual.(rightName));
    if leftGiven && ~rightGiven
        manual.(rightName) = mirror_y(as_point(manual.(leftName), leftName));
    elseif rightGiven && ~leftGiven
        manual.(leftName) = mirror_y(as_point(manual.(rightName), rightName));
    end
end
end

function out = merge_defaults(defaults, overrides)
out = defaults;
if isempty(overrides)
    return;
end
names = fieldnames(overrides);
for k = 1:numel(names)
    name = names{k};
    if isfield(out, name) && isstruct(out.(name)) && isstruct(overrides.(name))
        out.(name) = merge_defaults(out.(name), overrides.(name));
    else
        out.(name) = overrides.(name);
    end
end
end

function validate_parameters(p)
validate_scalar('p.wheelbase', p.wheelbase, true);
validate_scalar('p.geometryTolerance', p.geometryTolerance, true);
validate_axle('p.front', p.front);
validate_axle('p.rear', p.rear);
end

function validate_axle(name, a)
validate_scalar([name '.axleX'], a.axleX, false);
validate_scalar([name '.trackWidth'], a.trackWidth, true);
validate_scalar([name '.tireDiameter'], a.tireDiameter, true);
validate_scalar([name '.scrubRadius'], a.scrubRadius, false);
validate_scalar([name '.kpiDeg'], a.kpiDeg, false);
validate_scalar([name '.casterDeg'], a.casterDeg, false);
validate_scalar([name '.steeringAxisGroundXOffset'], a.steeringAxisGroundXOffset, false);
validate_scalar([name '.rollCenterHeight'], a.rollCenterHeight, false);
validate_scalar([name '.instantCenterY'], a.instantCenterY, false);
validate_scalar([name '.lowerBallJointHeight'], a.lowerBallJointHeight, true);
validate_scalar([name '.upperBallJointHeight'], a.upperBallJointHeight, true);
validate_scalar([name '.lowerInboardY'], a.lowerInboardY, true);
validate_scalar([name '.upperInboardY'], a.upperInboardY, true);
validate_scalar([name '.lowerInboardXOffset'], a.lowerInboardXOffset, false);
validate_scalar([name '.upperInboardXOffset'], a.upperInboardXOffset, false);
validate_scalar([name '.lowerArmHalfForeAft'], a.lowerArmHalfForeAft, true);
validate_scalar([name '.upperArmHalfForeAft'], a.upperArmHalfForeAft, true);
validate_scalar([name '.lowerInboardZDeltaForeMinusAft'], a.lowerInboardZDeltaForeMinusAft, false);
validate_scalar([name '.upperInboardZDeltaForeMinusAft'], a.upperInboardZDeltaForeMinusAft, false);
validate_fraction([name '.directMountFractionFromLBJ'], a.directMountFractionFromLBJ);
validate_fraction([name '.pushrodMountFractionFromUBJ'], a.pushrodMountFractionFromUBJ);
validate_scalar([name '.rockerPushrodArmLength'], a.rockerPushrodArmLength, true);
validate_scalar([name '.rockerDamperArmLength'], a.rockerDamperArmLength, true);
validate_scalar([name '.rockerArbArmLength'], a.rockerArbArmLength, true);
validate_scalar([name '.rockerPushrodAngleDeg'], a.rockerPushrodAngleDeg, false);
validate_scalar([name '.rockerDamperAngleDeg'], a.rockerDamperAngleDeg, false);
validate_scalar([name '.rockerArbAngleDeg'], a.rockerArbAngleDeg, false);
if a.upperBallJointHeight <= a.lowerBallJointHeight
    error('%s.upperBallJointHeight must be greater than lowerBallJointHeight.', name);
end
if abs(a.instantCenterY - a.trackWidth / 2) < 1.0e-9
    error('%s.instantCenterY cannot equal the left contact-patch y coordinate.', name);
end
end

function validate_fraction(name, value)
validate_scalar(name, value, false);
if value < 0 || value > 1
    error('%s must be between 0 and 1 inclusive.', name);
end
end

function validate_scalar(name, value, mustBePositive)
if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value)
    error('%s must be a finite numeric scalar.', name);
end
if mustBePositive && value <= 0
    error('%s must be positive.', name);
end
end

function pt = as_point(value, name)
if ~isnumeric(value) || numel(value) ~= 3 || any(~isfinite(value(:)))
    error('%s must be a finite numeric 1x3 point.', name);
end
pt = reshape(value, 1, 3);
end

function v = unit_vector(v)
n = norm(v);
if n < 1.0e-12
    error('Cannot normalize a zero-length vector.');
end
v = v / n;
end

function p = mirror_y(p)
p = as_point(p, 'mirror input');
p(2) = -p(2);
end

function name = side_token(side)
if side > 0
    name = 'Left';
else
    name = 'Right';
end
end

function name = side_field(side)
if side > 0
    name = 'left';
else
    name = 'right';
end
end

function out = title_case(in)
out = [upper(in(1)), lower(in(2:end))];
end