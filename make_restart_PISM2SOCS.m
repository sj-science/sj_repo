function [] = make_restart_PISM2SOCS(cpl_year,src_sim_cse)
%cpl_year=2012;
%src_sim_cse ='his_85NE';

%% some general options
force_cluster_machine=4;
script_time = tic;
sav_data = 'on';
chk_plt = 0;
if ischar(cpl_year)
    cpl_year=str2double(cpl_year);
end
%% processing source options
prop = [1 2 5 6 7];
src_var_nme = {'temp';  'salt' ; 'ubar'; 'vbar';'u';'v';'zeta'};
src_grd_typ = {'r';    'r';    'u';    'v';   'u';  'v';  'r'};
bog_val = [-5 80; 0 50; -5 5; -5 5; -5 5; -5 5; -10 10];

%% processing / target options
trg_var_nme = {'temp';  'salt' ; 'ubar'; 'vbar';'u';'v';'zeta'};
trg_grd_typ = {'r';    'r';    'u';    'v';   'u';  'v';  'r'};

add_src_layer = 'on';
add_src_layer_no = [5 6]; % 5:bottom 6:surface (doublecheck what the directions mean)

%% inputs -----------------------------------------------------------------
% files will be created in a local folder and then copied by the coupler
%% 1 retrieve and check target ini file name

trg_ini_fnm = ['027_ini_' read_PISM2SOCS_input('ROMS_TRG_INI_CSE') '.nc' ];

% check name of inifile
chk1 = isnumber(trg_ini_fnm(1:3));
chk2 = strcmpi(trg_ini_fnm(5:7),'ini');
chk3 = strcmpi(trg_ini_fnm(end-2:end),'.nc');
if ~(allel(chk1,[1 1 1])==1 && chk2==1 && chk3==1)
    disp('target ini file name is not consistent with conventions')
    return
end
% retrieve sim number
trg_sim_num = [trg_ini_fnm(1:3)];

%% 2 define local target path for ini file
if force_cluster_machine==4
    [~, trg_pre_pth, ~, ~, trg_pro_pth, ~] = set_machine_path(trg_sim_num,4);
else
    [~, trg_pre_pth, ~, ~, trg_pro_pth, ~] = set_machine_path(trg_sim_num);
end

trg_ini_pth = realpath(trg_pre_pth);
trg_ini_fle = [trg_ini_pth '/' trg_ini_fnm];


%% 3 retrieve and check target grid file
trg_grd_cse = read_PISM2SOCS_input('ROMS_TRG_GRD_CSE');
%trg_grd_cse='SOCS85NE_initial_2006'

trg_grd_fnm = ['027_grd_' trg_grd_cse '.nc'];
trg_grd_fle = [realpath(trg_pre_pth) '/' trg_grd_fnm];

if ~exist('trg_grd_fle','var')
    disp(['trg_grd_fle does not exist.'])
    return
end

chk1 = isnumber(trg_grd_fnm(1:3));
chk2 = strcmpi(trg_grd_fnm(5:7),'grd');
chk3 = strcmpi(trg_grd_fnm(end-2:end),'.nc');
if ~(allel(chk1,[1 1 1])==1 && chk2==1 && chk3==1)
    disp('target grid file name is not consistent with conventions')
    return
end

tmp = [trg_grd_fnm(1:3)];
if ~strcmpi(tmp,trg_sim_num)
    disp('target grid sim numbr inconsistent with ini file sim numbr')
    return
end
clear tmp

%% 3 source file

src_out_pth = realpath(read_PISM2SOCS_input('SOCS_SRC_OUT_PTH'));

% source data sim numbr, case, path
if ~strcmpi(src_out_pth(end),'/')
    src_out_pth = [src_out_pth '/'];
end
% aiming to find a restart file
data_info = data_case_V05([src_out_pth],{'time','tim'},src_var_nme(prop(:)),src_sim_cse);

src_sim_fnm = char(data_info.fle_nmes(1));
src_sim_num = src_sim_fnm(1:3); clear src_sim_fnm


disp('---------------------------------------------------------------------')
disp(['make_restart_PISM2SOCS:'])
disp(['source case    : ' src_sim_cse])
disp(['source path    : ' src_out_pth])
disp(['target grd file: ' trg_grd_fle])
disp(['target ini file: ' trg_ini_fle])


%% check input end ----------------------------------------------

%trg_ref_dte = data_info.src_ref_dte;  % target reference date same as source
trg_ref_dte = read_PISM2SOCS_input('SOCS_REF_DTE');

%% save matlab code (seems not working in runtime enviro)
%scr_pth = [trg_pre_pth 'scripts/arc/' ];
%g = save_script([realpath(read_PISM2SOCS_input('LOG_FLE_PTH')) '/fill_ini_script_' trg_sim_num '_' trg_ini_cse],mfilename('fullpath'));

N_trg_frm = 1; % one field target field (cut out later, not really needed anymore)

%% retrieve coordinates of target grid
if exist([trg_pre_pth 'pre_data/mapping_' trg_grd_cse '.mat'],'file')==2
    trg_zco_fle = [trg_pre_pth 'pre_data/mapping_' trg_grd_cse '.mat'];
elseif exist([trg_pro_pth 'pro_data/mapping_' trg_grd_cse '.mat'],'file')==2
    trg_zco_fle = [trg_pro_pth 'pro_data/mapping_' trg_grd_cse '.mat'];
else
    disp(['Cant spot the mapping file for target grid case ' trg_grd_cse ' ! STOP'])
    return
end
load(trg_zco_fle, 'trg_grd');

mask_rho = trg_grd.mask_rho;
mask_rho(isnan(mask_rho)) = 1; % ocean is 1 land is zero
mask_rho(mask_rho == 0) = NaN;

mask_u = trg_grd.mask_u;
mask_u(isnan(mask_u)) = 1; % ocean is 1 land is zero
mask_u(mask_u == 0) = NaN;

mask_v = trg_grd.mask_v;
mask_v(isnan(mask_v)) = 1; % ocean is 1 land is zero
mask_v(mask_v == 0) = NaN;

trg_grd.h_u = (trg_grd.h(:,1:end-1)+trg_grd.h(:,2:end))/2;
trg_grd.h_v = (trg_grd.h(1:end-1,:)+trg_grd.h(2:end,:))/2;
trg_grd.zice_u = (trg_grd.zice(:,1:end-1)+trg_grd.zice(:,2:end))/2;
trg_grd.zice_v = (trg_grd.zice(1:end-1,:)+trg_grd.zice(2:end,:))/2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load value, extrapolate, interpolate, fill nc file

disp(['************************************************************************'])
disp(['*                  retrieve source fields from model output            *'])
disp(['------------------------------------------------------------------------'])

src_files   = data_info.fle_nmes;
NC_SRC_TIME = data_info.src_tme;
NC_IND_STMP = data_info.nct_ind;
NC_HIS_STMP = data_info.fle_ind;
src_dpyr    = data_info.src_dpyr;
src_ref_dte = data_info.src_ref_dte;
src_tme_unt = data_info.src_tme_unt;

SRC2TRG_IND = zeros(size(NC_IND_STMP));
src_per_trg = zeros(size(NC_IND_STMP));

% conversion of time to days was already done by data_case
% since src time will not be read again we need to change unit here,
% according to the conversion inside data_case
src_tme_unt = 'd';

% establish month and year index
[tmp, ti] = calendar_index(NC_SRC_TIME,src_dpyr,src_ref_dte);
% select time record that is closest to the completed year=cpl_year
[~,closest_entry]=min(abs(ti.y-(cpl_year+1)));

%[~,latest_entry] = max(NC_SRC_TIME);
disp(['--   nc record closest to completed year ' num2str(cpl_year) ' is ' num2str(closest_entry)])
disp(['--   use closest entry for ini file: ' datestr(tmp(closest_entry,:))])

clear tmp

SRC2TRG_IND(closest_entry)=1;

for t_trg=1:N_trg_frm
    src_per_trg(t_trg) = sumsum(SRC2TRG_IND==t_trg);
end
clear t_trg


%% obtain full coordinate set of source grid (always needs editing according to source field)
% attain source grid file name, sim number and case (path will be determined later)
% this is a bit of a hack because setting SRC2TRG_IND==1 assumes
% N_trg_frm==1
% it needs to happen after data case because the source grid depends on
% which of the source files is used
i_fle=NC_HIS_STMP(SRC2TRG_IND==1);

[~, src_grd_fnm] = separate_file_path(ncreadatt([src_out_pth char(src_files(i_fle))],'/','grd_file'));
%src_grd_fnm = ncreadatt([src_out_pth char(data_info.fle_nmes(1))],'/','grd_file');
src_grd_num = src_grd_fnm(1:3);
src_grd_cse = src_grd_fnm(9:end-3);

% define local path paths
if force_cluster_machine==4
    %    [~, ~, ~, raw_path] = set_machine_path(src_sim_num,4);
    [~,src_pre_pth,~,~,src_pro_pth,~] = set_machine_path(src_grd_num,4);
else
    %    [~, ~, ~, raw_path] = set_machine_path(src_sim_num);
    [~,src_pre_pth,~,~,src_pro_pth,~] = set_machine_path(src_grd_num);
end
% retrieve from source file
if exist([src_pre_pth 'pre_data/mapping_' src_grd_cse '.mat'],'file')
    src_zco_fle = [src_pre_pth 'pre_data/mapping_' src_grd_cse '.mat'];
elseif exist([src_pro_pth 'pro_data/mapping_' src_grd_cse '.mat'],'file')
    src_zco_fle = [src_pro_pth 'pro_data/mapping_' src_grd_cse '.mat'];
else
    disp(['Cant spot the mapping file for source grid case ' src_grd_cse ' ! STOP'])
    return
end


tmp = load(src_zco_fle,'trg_grd');
SRC_GRD = tmp.trg_grd;
clear tmp

% assign different types of grid to make available for processing below

src_grd.z_r = SRC_GRD.z_r;
src_grd.z_r = permute(src_grd.z_r,[2 3 1]);
src_grd.h_r = SRC_GRD.h;
src_grd.zice_r = SRC_GRD.zice;
src_grd.mask_rho = SRC_GRD.mask_rho;

src_grd.z_u= SRC_GRD.z_u;
src_grd.z_u = permute(src_grd.z_u,[2 3 1]);
src_grd.h_u = (SRC_GRD.h(:,1:end-1)+SRC_GRD.h(:,2:end))/2;
src_grd.zice_u = (SRC_GRD.zice(:,1:end-1)+SRC_GRD.zice(:,2:end))/2;
src_grd.mask_u = SRC_GRD.mask_u;

src_grd.z_v = SRC_GRD.z_v;
src_grd.z_v = permute(src_grd.z_v,[2 3 1]);
src_grd.h_v = (SRC_GRD.h(1:end-1,:)+SRC_GRD.h(2:end,:))/2;
src_grd.zice_v = (SRC_GRD.zice(1:end-1,:)+SRC_GRD.zice(2:end,:))/2;
src_grd.mask_v = SRC_GRD.mask_v;

src_grd.N = SRC_GRD.N;

clear SRC_GRD



%% create ini file
if exist(trg_ini_fle,'file')
    disp(['removing existing ' trg_ini_fle])
    delete(trg_ini_fle)
end
create_ROMS_ini(trg_ini_fle,size(mask_rho,2),size(mask_rho,1),trg_grd.N,1,'single');


%% loop over properties
for j=1:size(prop,2)
    disp(['------------------------------------------------------------------------'])
    disp(['processing ' char(src_var_nme(prop(j))) ])
    disp(['source GRD file: ' src_grd_fnm])
    disp(['source OUT file: ' char(src_files(end))])
    disp(['target GRD file: ' trg_grd_fnm])
    disp(['target INI file: ' trg_ini_fnm])
    disp(['------------------------------------------------------------------------'])
    
    
    %% assign full target and source grids according to sub grid type
    % note:  three different staggered grids involved - rho / u / v
    % note 1: make sure source depth is negative
    % note 2: depth index increases with increasing depth z(1)=surface
    % z(end)=bottom
    
    
    switch char(trg_grd_typ(prop(j))) %target
        case 'r'  % rho grid coordinates
            Z_TRG   =  permute(trg_grd.z_r,[2,3,1]);
            MSK_TRG =  trg_grd.mask_rho;
            H_TRG = trg_grd.h;
            I_TRG = trg_grd.zice;
        case 'u'  % u grid
            Z_TRG   =  permute(trg_grd.z_u,[2,3,1]);
            MSK_TRG =  trg_grd.mask_u;
            tmp = trg_grd.angle;
            H_TRG = trg_grd.h_u;
            I_TRG = trg_grd.zice_u;
        case 'v'  % v grid
            Z_TRG   =  permute(trg_grd.z_v,[2,3,1]);
            MSK_TRG =  trg_grd.mask_v;
            H_TRG = trg_grd.h_v;
            I_TRG = trg_grd.zice_v;
    end
    
    switch char(src_grd_typ(prop(j))) % source
        case 'r'  % rho grid coordinates
            Z_SRC   =  src_grd.z_r;
            MSK_SRC =  src_grd.mask_rho;
            H_SRC = src_grd.h_r;
            I_SRC = src_grd.zice_r;
        case 'u'  % u grid
            Z_SRC   =  src_grd.z_u;
            MSK_SRC =  src_grd.mask_u;
            H_SRC = src_grd.h_u;
            I_SRC = src_grd.zice_u;
        case 'v'  % v grid
            Z_SRC   =  src_grd.z_v;
            MSK_SRC =  src_grd.mask_v;
            H_SRC = src_grd.h_v;
            I_SRC = src_grd.zice_v;
    end
    if ~strcmp(src_grd_fnm,trg_grd_fnm)
        % reduce DOM_TRG to those spots that changed bathymetry
        if src_grd.N~=trg_grd.N
            DOM_TRG = MSK_TRG==1;
        else
            DOM_TRG = MSK_TRG==1&(I_SRC~=I_TRG|H_SRC~=H_TRG|MSK_TRG~=MSK_SRC);
        end
        [dom_iet,dom_ixi] = find_crop_index(DOM_TRG,0,'exp');
        
        %% retrieve full time series of property
        % note, the loop over all boundaries starts after the loading step here
        % implicitly assumes that here all lateral source points are loaded
        % If memory is an issue (i.e. large .nc model outputs e.g. source==4)
        % the loading is recommended to be carried out inside the bry_cnt loop
        % retrieve information about property
        if any([1 2 5 6]==prop(j))
            vardim=3;
        else
            vardim=2;
        end
        
        
        %% 2nd loop over west/north/east/south
        
        % welcome message
        disp('=================================================================================')
        disp(['establish ini condition ' char(src_var_nme(prop(j))) ' ']);
        disp('---------------------------------------------------------------------------------')
        % initialize vars
        
        if ~exist('TIME_TRG','var')
            disp('--   initialize time_trg ')
            time_trg = NaN(1,N_trg_frm);
        else
            disp('--   grab time_trg from manually pre defined TIME_TRG')
            time_trg = C2M(TIME_TRG(prop(j)))';
        end
        
        msk_trg = MSK_TRG(dom_iet(1):dom_iet(2),dom_ixi(1):dom_ixi(2));
        z_trg = Z_TRG(dom_iet(1):dom_iet(2),dom_ixi(1):dom_ixi(2),:);
        dom_trg = DOM_TRG(dom_iet(1):dom_iet(2),dom_ixi(1):dom_ixi(2));
        
        
        %% adjust trg coordinates according to type of bc field, expand to full 3D
        % the output form of xxx_trg [lat/lon depth] / input is [1 lat/lon]
        
        %     % turn mask into useful values ocean=1 land=NaN
        %     msk_trg(isnan(msk_trg)) = 1;
        %     msk_trg(msk_trg==0) = NaN;
        
        
        %% initialize time shot collector
        if vardim==3
            bryc3 = zeros([trg_grd.N sumsum(dom_trg==1)]);
        elseif vardim==2
            bryc2 = zeros([1 sumsum(dom_trg==1)]);
        end
        
        %% crop src dom/msk/z according to bounds
        if vardim==3
            z_src = Z_SRC(dom_iet(1):dom_iet(2),dom_ixi(1):dom_ixi(2),:);
        end
        msk_src = MSK_SRC(dom_iet(1):dom_iet(2),dom_ixi(1):dom_ixi(2));
        
        
        %% attain sizes of target
        n_trg_eta = size(msk_trg,1);
        n_trg_xi = size(msk_trg,2);
        
        %% add layer to the top and bottom
        if strcmpi(add_src_layer, 'on') && vardim==3
            %% add vertical layer (needs to comply with layer-add of bc_src)
            
            if any(add_src_layer_no==5) % bottom
                disp('--   adding bottom layer to z')
                z_src = add_layer(z_src,3,2,-10000,'+'); %200,'+'); % top
            end
            if any(add_src_layer_no==6) % surface
                disp('--   adding surface layer to z')
                z_src = add_layer(z_src,3,1,1.5); % bottom
                % the uppermost (added) source sigma level is set to 1.5 m to
                % make sure 0 m as expected minimum target depth is included
                % in the vertical interpolation
            end
            
        end
        
        
        %% definition end of src coordinates
        %==================================================================
        
        %% try reducing the number of horizontal points to speed up interpolation later
        map_fld_src = zeros(size(msk_src));
        id1D_fld_src = find(msk_src==1&dom_trg==1);
        
        map_fld_src(id1D_fld_src)=1;
        
        
        %% linearize and reduce input coordinates
        disp(['--   reducing source coordinates by ' num2str(round_dec(1-sumsum(map_fld_src)/numel(msk_src),6)*100) '%'])
        
        if vardim==3
            %  z_src_=double(z_src(map_fld_src==1));
        end
        
        
        for t_trg=1:N_trg_frm    % loop over all target intervals
            frame_time = tic;
            disp_txt = {};
            disp(['--   target time stamp: ' num2str(t_trg) ' of ' num2str(N_trg_frm) ' | ' char(src_var_nme(prop(j))) ' --------------']);
            
            % collect source stamps and average for each target stamp
            nc_ind_stmp=NC_IND_STMP(SRC2TRG_IND==t_trg);
            nc_his_stmp=NC_HIS_STMP(SRC2TRG_IND==t_trg);
            nc_src_time=NC_SRC_TIME(SRC2TRG_IND==t_trg);
            N_src_per_trg = length(nc_ind_stmp);
            i_nc=1;
            time_read = tic;
            
            while i_nc<=N_src_per_trg
                [nc_c, nc_inc]  = nc_monotonic(nc_ind_stmp(i_nc:end),nc_his_stmp(i_nc:end));
                nc_s            = nc_ind_stmp(i_nc);
                i_src_fle       = nc_his_stmp(i_nc);
                disp(['--   nc index s/c/strd:  ' num2str([nc_s nc_c nc_inc]) ]);
                disp(['     ' C2M(src_files(nc_his_stmp(i_nc))) ' --------']);
                %% retrieve data containers or source file
                % nc-output: [lon lat (depth) time] permute to: [lat lon (depth) time]
                
                if vardim==3
                    his_file = [src_out_pth C2M(src_files(i_src_fle))];
                    
                    if isfinite(strfind(his_file,'_rst_'))
                        tmp_src = squeeze(ncread(his_file,char(src_var_nme(prop(j))),[dom_ixi(1),dom_iet(1),1,1,nc_s],[dom_ixi(2)-dom_ixi(1)+1,dom_iet(2)-dom_iet(1)+1,Inf,1,nc_c],[1,1,1,1,nc_inc]));
                    else
                        tmp_src = ncread(his_file,char(src_var_nme(prop(j))),[dom_ixi(1),dom_iet(1),1,nc_s],[dom_ixi(2)-dom_ixi(1)+1,dom_iet(2)-dom_iet(1)+1,Inf,nc_c],[1,1,1,nc_inc]);
                    end
                    
                    tmp_src = permute(tmp_src,[2,1,3,4]);
                    
                    % initialize field
                    if i_nc==1
                        bc_src = zeros([size(tmp_src,1) size(tmp_src,2) size(tmp_src,3)]);
                    end
                    % implicit averaging
                    bc_src = bc_src+sum(tmp_src,4)/N_src_per_trg;
                    
                elseif vardim==2
                    his_file = [src_out_pth C2M(src_files(i_src_fle))];
                    
                    if isfinite(strfind(his_file,'_rst_'))
                        tmp_src = squeeze(ncread(his_file,char(src_var_nme(prop(j))),[dom_ixi(1),dom_iet(1),1,nc_s],[dom_ixi(2)-dom_ixi(1)+1,dom_iet(2)-dom_iet(1)+1,1,nc_c],[1,1,1,nc_inc]));
                    else
                        tmp_src = ncread(his_file,char(src_var_nme(prop(j))),[dom_ixi(1),dom_iet(1),nc_s],[dom_ixi(2)-dom_ixi(1)+1,dom_iet(2)-dom_iet(1)+1,nc_c],[1,1,nc_inc]);
                    end
                    tmp_src = permute(tmp_src,[2,1,3]);
                    
                    
                    % initialize field
                    if i_nc==1
                        bc_src = zeros([size(tmp_src,1) size(tmp_src,2)]);
                    end
                    % implicit averaging
                    bc_src = bc_src+sum(tmp_src,3)/N_src_per_trg;
                end
                bc_src = squeeze(bc_src);
                i_nc = i_nc+nc_c;
                clear tmp_src
                
            end
            
            (['--   reading source from nc file: ' num2str(toc(time_read)) 's']);

            %% time
            time_trg(t_trg) = mean(nc_src_time);
            
            
            %% add vertical layers to source
            if strcmpi(add_src_layer, 'on') && vardim==3
                tic
                if any(add_src_layer_no==5) % bottom
                    disp('--   adding bottom layer to bc');
                    %disp('adding bottom layer to bc')
                    bc_src = add_layer(bc_src,3,2,2); % add layer at the top witch is a copy of adjacent layer
                end
                if any(add_src_layer_no==6) % surface
                    disp('--   adding surface layer to bc');
                    %disp('adding surface layer to bc')
                    bc_src = add_layer(bc_src,3,1,2); % add layer at the bottom witch is a copy of adjacent layer
                end
                disp(['--   adding source layers: ' num2str(round_dec(toc,2)) 's']);
            end
            
            %--------------------------------------------------------------
            %%  interpolate spots that only changed zice or h
            % linearize source fields
            bc_src_ = double(bc_src(map_fld_src==1));
            %bc_src = permute(bc_src,[3 1 2]);
            NaN_in_src = sumsum(isnan(bc_src_));
            bogus_values = sumsum(bc_src<bog_val(prop(j),1)|bc_src>bog_val(prop(j),2));
            if t_trg==1 && (NaN_in_src>0||bogus_values>0)
                %disp(char(disp_txt))
                disp(['--   detected ' num2str(NaN_in_src) ' NaNs and ' num2str(bogus_values) ' in linearized source field (' num2str(100*NaN_in_src/numel(bc_src_)) '%). This might cause problems with the interpolant.']);
            end
            minmax_src= [];
            switch vardim
                case 3
                    tic
                    bc_src = permute(bc_src,[3 1 2]);
                    z_src = permute(z_src,[3 1 2]);
                    z_trg = permute(z_trg,[3 1 2]);
                    bc = NaN([trg_grd.N size(dom_trg)]);
                    %                 for id=1:length(id1D_fld_src)
                    %                     minmax_src = minmax([minmax(bc_src(:,id1D_fld_src(id))) minmax_src ]);
                    %                     bc(:,id1D_fld_src(id)) = interp1(z_src(:,id1D_fld_src(id)),bc_src(:,id1D_fld_src(id)),z_trg(:,id1D_fld_src(id)));
                    %                 end
                    
                    % try parfor
                    tmp_bc = bc(:,id1D_fld_src);
                    tmp_z_src = z_src(:,id1D_fld_src);
                    tmp_bc_src = bc_src(:,id1D_fld_src);
                    tmp_z_trg = z_trg(:,id1D_fld_src);
                    minmax_src = minmax([minmax(bc_src(:,id1D_fld_src))]);
                    parfor id=1:length(id1D_fld_src)
                        tmp_bc(:,id) = interp1(tmp_z_src(:,id),tmp_bc_src(:,id),tmp_z_trg(:,id));
                    end
                    bc(:,id1D_fld_src)=tmp_bc;
                    clear tmp_bc tmp_z_src tmp_bc_src tmp_z_trg
                    disp(['--   vertical interpolation: ' num2str(round_dec(toc,2)) 's']);
                case 2
                    bc = NaN([size(dom_trg)]);
                    for id=1:length(id1D_fld_src)
                        bc(id1D_fld_src(id)) = bc_src(id1D_fld_src(id));
                    end
                    minmax_src = minmax([minmax(bc_src(id1D_fld_src(id))) minmax_src ]);
                    disp(['--   no vertical interpolation needed for 2D field']);
            end
            
            
            %% extrapolate into new ocean areas that were not ocean before
            % strategy is to isolate NaN patches, crop them,
            % extrapolate individually (this is memory conserving)
            nan_trg_msk = (msk_src==0&msk_trg==1);
            
            % create index that will be saved in ini file
            proc_msk = msk_trg; % ocean=1,land=0
            proc_msk(dom_trg==1)=2; % interpolation=2
            proc_msk(nan_trg_msk==1)=3; % extrapolation=3
            if chk_plt==1&&j==1
                figure;imagesc(proc_msk);
                title('Areas: 2:interpolated 3:extrapolated')
                caxis([0 3]);
            end
            tmp = MSK_TRG;
            tmp(dom_iet(1):dom_iet(2),dom_ixi(1):dom_ixi(2))=proc_msk;
            proc_msk=tmp;clear tmp
            
            
            [c,d] = islands(nan_trg_msk);
            d(d(:,3)==0,:)=[];
            % deal with one cell spots i.e. c==0
            one_cll_ind = find(c==0);
            max_c = max(d(:,1));
            for id=1:length(one_cll_ind)
                reg_id = max_c+id;
                if nan_trg_msk(one_cll_ind(id))==1
                    d =  [d;[reg_id 1 1]];
                    c(one_cll_ind(id))=reg_id;
                end
            end
            clear nan_trg_msk
            reg_id_of_extrap_error = []; % collects the erroneous extrapolation regions
            for id=1:size(d,1)
                if ~any([3 4 5 6]==prop(j))||id==1
                    disp('------------------------------------------------------------------------------------')
                end
                
                % define extrapolation target (should be one connected region)
                % 1:extrapolated/0:not to be extrapolated
                exp_trg = double(c==d(id,1));
                
                % find crop index for that area
                xi = [max([1,find(sum(exp_trg,1)~=0,1,'first')-1]) min([n_trg_xi find(sum(exp_trg,1)~=0,1,'last')+1])];
                et = [max([1,find(sum(exp_trg,2)~=0,1,'first')-1]) min([n_trg_eta find(sum(exp_trg,2)~=0,1,'last')+1])];
                
                % estimate max extrapolation rounds
                ext_n = max([diff(xi) diff(et)]);
                
                % crop extrapolation mask and 1:ocean/0:land mask to minimal size
                exp_trg = exp_trg(et(1):et(2),xi(1):xi(2));
                rho_msk = msk_trg(et(1):et(2),xi(1):xi(2));
                
                % exclude land areas, slightly change meaning
                % -1:no source/0:source for extrapolation/1:to be extrapolated
                exp_trg(rho_msk==0)=-1;
                
                % find the 1D index of cells inside extrapolation area
                id_exp_trg = find(exp_trg==1);
                
                % create halo of valid points around exp_trg, width one cell
                % because of exp_trg=-1/0/1 exp_hlo is zero in land area
                [~,exp_hlo] = make_index_boundary_V01(exp_trg,[0 1],0.0,1);
                
                % exclude the actual extrapolation area from halo
                exp_hlo(exp_trg==1)=0;
                ind_hlo = find(exp_hlo==1);
                
                % reduce the valid extrapolation sources to points within halo
                exp_trg(exp_trg==0&exp_hlo==0)=-1;
                
                if ~any([3 4 5 6]==prop(j))
                    disp(['--   extrapolating region ' num2str(id) '/' num2str(size(d,1)) ...
                        '  size ' num2str(sumsum(exp_trg==1))])
                elseif id==1
                    disp(['--   no extrapolating for velocities. '])
                end
                
                tic0 = tic;
                if vardim==3
                    %% s2z vertical interpolation in order to sensibly extrapolate
                    
                    z_s = z_src(:,et(1):et(2),xi(1):xi(2));
                    z_t = z_trg(:,et(1):et(2),xi(1):xi(2));
                    b_s = bc_src(:,et(1):et(2),xi(1):xi(2));
                    
                    dlvl = [floor(minmin(z_t(:,exp_trg==1))):2:ceil(maxmax(z_t(:,exp_trg==1))+1)]';
                    
                    b_s2z = NaN([length(dlvl) size(exp_trg)]);
                    
                    for ie=1:length(ind_hlo)
                        b_s2z(:,ind_hlo(ie)) = interp1(z_s(:,ind_hlo(ie)),b_s(:,ind_hlo(ie)),dlvl);
                    end
                    
                    clear z_s b_s
                elseif vardim==2
                    dlvl = 0;
                    b_s = bc_src(et(1):et(2),xi(1):xi(2));
                    b_s2z = NaN([size(exp_trg)]);
                    b_s2z(ind_hlo) = b_s(ind_hlo);
                end
                t_interp1 = toc(tic0);
                
                %% extrapolate
                tic1 =  tic;
                if any([3 4 5 6]==prop(j))
                    %disp('No extrapolation for velocities. Set to zero instead')
                    b_s2z = permute(b_s2z,[2 3 1]);
                    b_s2z(repmat(exp_trg,[1 1 length(dlvl)])==1)=0;
                else
                    
                    if vardim==3
                        tmp = permute(b_s2z,[2 3 1]);
                        
                        minmax_bef_extr = ((minmax(tmp(repmat(exp_trg,[1 1 length(dlvl)])==0)))); % halo source area
                        
                        b_s2z = nan_extrap_V03(tmp,ext_n*2,'h','noplt',3,int8(repmat(exp_trg,[1 1 length(dlvl)])));
                        
                        minmax_aft_extr = ((minmax(b_s2z(repmat(exp_trg,[1 1 length(dlvl)])==1)))); % extrapolated area
                        
                        % check for extrapolation error
                        if minmax_aft_extr(1)<minmax_bef_extr(1) || minmax_aft_extr(2)>minmax_bef_extr(2)
                            disp(['before: ' num2str(minmax_bef_extr)])
                            disp(['after:  ' num2str(minmax_aft_extr)])
                            reg_id_of_extrap_error(length(reg_id_of_extrap_error)+1)=id;
                        end
                        clear tmp minmax_aft_extr minmax_bef_extr
                    elseif vardim==2
                        b_s2z = nan_extrap_V03(b_s2z,ext_n*2,'h','noplt',3,exp_trg);
                    end
                end
                % check if it needs more extrapolation is needed
                for in=1:size(b_s2z,3)
                    tmp = b_s2z(:,:,in);
                    if any(isnan(tmp(exp_trg==1)))==1
                        disp(['NaN left after extrapolation at depth index ' num2str(in)])
                        return
                    end
                end
                t_extrap = toc(tic1);
                
                if vardim ==3
                    b_s2z = permute(b_s2z,[3 1 2]);
                    
                    
                    %% z2s vertical interpolation onto target grid
                    tic2 = tic;
                    b_z2s = NaN([trg_grd.N size(rho_msk)]);
                    for ie=1:length(id_exp_trg)
                        b_z2s(:,id_exp_trg(ie)) = interp1(dlvl,b_s2z(:,id_exp_trg(ie)),z_t(:,id_exp_trg(ie)));
                    end
                    t_interp2 = toc(tic2);
                    b_t = bc(:,et(1):et(2),xi(1):xi(2));
                    b_t(:,id_exp_trg) = b_z2s(:,id_exp_trg);
                    bc(:,et(1):et(2),xi(1):xi(2)) = b_t;
                    clear b_t b_s2z b_z2s z_t b_s dlvl
                    tmp_txt =  ['interpolation: ' num2str(round_dec(t_interp1+t_interp2,1)) '[s] '];
                elseif vardim==2
                    b_z2s = NaN([size(rho_msk)]);
                    b_z2s(id_exp_trg) = b_s2z(id_exp_trg);
                    b_t = bc(et(1):et(2),xi(1):xi(2));
                    b_t(id_exp_trg) = b_z2s(id_exp_trg);
                    bc(et(1):et(2),xi(1):xi(2)) = b_t;
                    clear b_t b_s2z b_z2s b_s dlvl
                    tmp_txt='';
                    
                end
                
                if ~any([3 4 5 6]==prop(j))
                    disp(['--   ' tmp_txt 'extrapolation: ' num2str(round_dec(t_extrap,1)) '[s]'])
                end
            end
            if length(reg_id_of_extrap_error)>1
                err_msk = zeros(size(c));
                for il=1:length(reg_id_of_extrap_error)
                    %err_msk(c==reg_id_of_extrap_error(il))=1;
                    err_msk(msk_trg==0)=-1;
                    err_msk(c==reg_id_of_extrap_error(il))=1;
                    
                end
                figure;imagesc(err_msk);title('regions with extrapolation errors')
                return
            else
                disp('--  no apparent extrapolation errors ')
            end
            disp('------------------------------------------------------------------------------------')
            
            %% collect time shots
            % ncwrite expects fields as [lat/lon (depth) time]
            if vardim==3
                bryc3(:,:) = bc(:,dom_trg==1);
            elseif vardim==2
                bryc2(:) = bc(dom_trg==1);
            end
            
            disp([' ini condition processing: ' num2str(toc(frame_time)) 's.']);
            disp('----------------------------------------------------------');
            disp(char(disp_txt))
        end
        
        % this renaming was neccessary for implementing parfor
        if vardim==3
            bryc = bryc3;
            clear bryc3
        elseif vardim==2
            bryc = bryc2;
            clear bryc2
        end
        
      %  minmax_trg = minmax(bryc(isfinite(bryc(:))));

    else
        disp(['source and target grids are identical'])
        disp('No interpolation and extrapolation needed')
        disp(['------------------------------------------------------------------------'])
        
        nc_src_time=NC_SRC_TIME(SRC2TRG_IND==1);
        nc_ind_stmp=NC_IND_STMP(SRC2TRG_IND==1);
        nc_his_stmp=NC_HIS_STMP(SRC2TRG_IND==1);
        nc_s            = nc_ind_stmp(1);
        i_src_fle       = nc_his_stmp(1);        
        nc_c=1;
        time_trg = mean(nc_src_time);
        % create index that will be saved in ini file
        proc_msk = MSK_TRG;
        if any([1 2 5 6]==prop(j))
            vardim=3;
        else
            vardim=2;
        end
        
        his_file = [src_out_pth C2M(src_files(i_src_fle))];
        
    end
    
    
    
    %% set time to reference start time
    % source time and therefore time_trg were already shifted to matlab
    % zero reference datenum([000-00-00 00:00:00])==0 in data_case_V02
    %  time_trg = time_stamp_converter(time_trg,{'d','d'},{'true','true'},[datenum(trg_ref_dte);0]);
    time_trg = time_stamp_converter(time_trg,['d','d'], ...
        {'true'; 'true'},{trg_ref_dte;src_ref_dte},[src_dpyr]);
    
  
    
    %% write time in nc file
    if strcmpi(sav_data, 'on')
        %% bad hack make first time stamp zero
        %time_trg(1)=0;
        ncwrite(trg_ini_fle,['ocean_time'],time_trg,[1]);
        
        if j==1
            % time
            if strcmpi(src_tme_unt,'d'); wrt_unt = 'days';
            elseif strcmpi(src_tme_unt,'s'); wrt_unt = 'seconds';
            elseif strcmpi(src_tme_unt,'m'); wrt_unt = 'months';
            elseif strcmpi(src_tme_unt,'y'); wrt_unt = 'years'; end
            ncwriteatt(trg_ini_fle,['ocean_time'],'units',[wrt_unt ' since ' datestr(trg_ref_dte,'yyyy-mm-dd HH:MM:ss')]);
            if src_dpyr==365; cal_unit='noleap';
            elseif src_dpyr>365; cal_unit='gregorian';
            elseif src_dpyr==360; cal_unit='360_day';
            end
            ncwriteatt(trg_ini_fle,['ocean_time'],'calendar',cal_unit);
            % bathymetry
            ncwrite(trg_ini_fle,'h',trg_grd.h')
            ncwrite(trg_ini_fle,'zice',trg_grd.zice')
            ncwrite(trg_ini_fle,'lat_rho',trg_grd.lat_rho')
            ncwrite(trg_ini_fle,'lon_rho',trg_grd.lon_rho')
            ncwrite(trg_ini_fle,'lat_u',trg_grd.lat_u')
            ncwrite(trg_ini_fle,'lon_u',trg_grd.lon_u')
            ncwrite(trg_ini_fle,'lat_v',trg_grd.lat_v')
            ncwrite(trg_ini_fle,'lon_v',trg_grd.lon_v')
            ncwrite(trg_ini_fle,'Cs_r',trg_grd.Cs_r)
            ncwrite(trg_ini_fle,'Cs_w',trg_grd.Cs_w)
            ncwrite(trg_ini_fle,'s_w',trg_grd.s_w)
            ncwrite(trg_ini_fle,'s_rho',trg_grd.sc_r)
            ncwrite(trg_ini_fle,'Vtransform',trg_grd.Vtransform)
            ncwrite(trg_ini_fle,'Vstretching',trg_grd.Vstretching)
            ncwrite(trg_ini_fle,'theta_s',trg_grd.theta_s)
            ncwrite(trg_ini_fle,'theta_b',trg_grd.theta_b)
            ncwrite(trg_ini_fle,'Tcline',trg_grd.Tcline)
            ncwrite(trg_ini_fle,'hc',trg_grd.hc)
            ncwrite(trg_ini_fle,'mask_rho',trg_grd.mask_rho')
            ncwrite(trg_ini_fle,'mask_ice',trg_grd.mask_ice')
            ncwrite(trg_ini_fle,'mask_u',trg_grd.mask_u')
            ncwrite(trg_ini_fle,'mask_v',trg_grd.mask_v')
            ncwrite(trg_ini_fle,'mask_ini_proc',proc_msk')
            disp('--  written time & grid data to ini file')
        end
        
        %% write data into nc file
        if vardim==2
            if src_grd.N==trg_grd.N || strcmp(src_grd_fnm,trg_grd_fnm)
                % insert the interpolated and extrapolated areas into the whole
                if isfinite(strfind(his_file,'_rst_'))
                    TRG_FLD = squeeze(ncread(his_file,char(trg_var_nme(prop(j))),[1 1 1 nc_s],[Inf Inf 1 nc_c]));
                else
                    TRG_FLD = squeeze(ncread(his_file,char(trg_var_nme(prop(j))),[1 1 nc_s],[Inf Inf nc_c]));
                end
            else
                TRG_FLD=zeros(size(ncread(trg_ini_fle,[char(trg_var_nme(prop(j)))])));
            end
            if ~strcmp(src_grd_fnm,trg_grd_fnm)
                trg_fld = TRG_FLD(dom_ixi(1):dom_ixi(2),dom_iet(1):dom_iet(2));
                trg_fld = trg_fld';
                trg_fld(dom_trg==1)=bryc;
                TRG_FLD(dom_ixi(1):dom_ixi(2),dom_iet(1):dom_iet(2)) = trg_fld';
                TRG_FLD(isnan(TRG_FLD))=0;
            else
               TRG_FLD(isnan(TRG_FLD))=0; 
            end
            ncwrite(trg_ini_fle,[char(trg_var_nme(prop(j)))],TRG_FLD,[1 1 1]);
            
        elseif vardim==3
            if src_grd.N==trg_grd.N || strcmp(src_grd_fnm,trg_grd_fnm)
                if isfinite(strfind(his_file,'_rst_'))
                    TRG_FLD = squeeze(ncread(his_file,char(trg_var_nme(prop(j))),[1 1 1 1 nc_s],[Inf Inf Inf 1 nc_c]));
                else
                    TRG_FLD = squeeze(ncread(his_file,char(trg_var_nme(prop(j))),[1 1 1 nc_s],[Inf Inf Inf nc_c]));
                end
            else
                TRG_FLD=zeros(size(ncread(trg_ini_fle,[char(trg_var_nme(prop(j)))])));
            end
            if ~strcmp(src_grd_fnm,trg_grd_fnm)
                trg_fld = TRG_FLD(dom_ixi(1):dom_ixi(2),dom_iet(1):dom_iet(2),:);
                for in=1:trg_grd.N
                    
                    tmp = trg_fld(:,:,in)';
                    
                    tmp(dom_trg==1) = bryc(in,:);
                    trg_fld(:,:,in) = tmp';
                end
                TRG_FLD(dom_ixi(1):dom_ixi(2),dom_iet(1):dom_iet(2),:) = trg_fld;
                TRG_FLD(isnan(TRG_FLD))=0;
            else
               TRG_FLD(isnan(TRG_FLD))=0;
            end

            ncwrite(trg_ini_fle,[char(trg_var_nme(prop(j)))],TRG_FLD,[1 1 1 1]);
        end
        minmax_trg = minmax(TRG_FLD(isfinite(TRG_FLD(:))));
        if strcmp(trg_grd_fnm,src_grd_fnm)
            minmax_src=minmax_trg;
        end
        disp(['--  source: ' num2str(minmax_src) ' target: ' num2str(minmax_trg)])
        disp(['--  ' char(trg_var_nme(prop(j)))  ' written into ini file. '])
        
        
    end
    
    disp(['============================================================'])
    
    if strcmpi(sav_data, 'on')
        
        disp(['--   ' num2str(N_trg_frm) ' initial fields processed and transferred from'])
        disp(char(src_files(unique(NC_HIS_STMP(SRC2TRG_IND~=0)))))
        
        disp(['to'])
        disp([trg_ini_fle])
        %        disp('target grid file:')
        %        disp([trg_grd_fle])
    end
end

if any(anyel([5 6],prop))
    if force_cluster_machine==4
        ini_u2ubar( trg_ini_fle,trg_grd_cse,trg_sim_num,4 );
    else
        ini_u2ubar( trg_ini_fle,trg_grd_cse,trg_sim_num);
    end
end

disp(['processing took ' num2str(toc(script_time)) 's'])
disp(['************************************************************'])


end
