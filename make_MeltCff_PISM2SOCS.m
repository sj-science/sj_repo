function []=make_MeltCff_PISM2SOCS(cpl_P2S_next)

%cpl_P2S_next=2015;

if ischar(cpl_P2S_next)
   cpl_P2S_next=str2double(cpl_P2S_next); 
end

script_timer = tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% set sim case & number, pathes, in- and output files
sim_numbr = ['027'];

grd_case = [read_PISM2SOCS_input('ROMS_TRG_GRD_CSE')];
%grd_case='85NE2011'
frc_case = [read_PISM2SOCS_input('ROMS_TRG_FRC_CSE')];
SOCS_SRC_CSE=[read_PISM2SOCS_input('SOCS_SRC_CSE')];

%grd_path = [read_PISM2SOCS_input('SOCS_TRG_INP_PTH')];
% read finished model year

% read length of coupling cycle
cpl_cycl= [read_PISM2SOCS_input('cpl_P2S_frq')];

% this should be consistent with the associated preprocessed forcing
trg_dpyr = 365.24;
force_cluster_machine=4; % set to 4 for mahuika cluster submission
force_reload_parpool=0; % how many rounds per reload?
parpool_loaded=0;
Ncpu=0;


% getting path sorted
if force_cluster_machine==4
    disp(' ')
    disp('[Submitting to mahuika cluster.]')
    disp('[Forcing machine paths to mahuika cluster.]')
    disp('')
    [roo_path, pre_path, inp_path, out_path, pro_path, tmp_path] = set_machine_path(sim_numbr,4);
else
    [roo_path, pre_path, inp_path, out_path, pro_path, tmp_path] = set_machine_path(sim_numbr);
end
%script_path = [pre_path 'scripts/'];
%script_path = realpath(read_PISM2SOCS_input('LOG_FLE_PTH'));

grd_file =  [pre_path sim_numbr '_grd_' grd_case '.nc'];



%% keywords:
% what data source - 21:ERA5 22:CNRM_CM5rcp85 23:NorESM1rcp26 24:MRI_ESM2 6:Tamura
n_src = 25;
dummy=0;
var_lst = {30 32 33 34 31}; %30 32 33 34 31
%30: thickness in PISM model
%32: ice mask from PISM model
%33: ice shelf mask from PISM model
%34: bedrock topography of PISM model
%31: melting coefficient PISM/ROMS
N_mem_split = [ 1 1 1 1 1]; % splits the processing into N_mem_split rounds to conserve memory
s_mem_split = [1 1 1 1 1];

dfl_lvl = 3;

% check input parameter
if numel(N_mem_split)<numel(var_lst) || numel(s_mem_split)<numel(var_lst)
   disp('not enough process splitting parameters provided. STOP') 
   return
end

if dummy~=0
   N_mem_split = ones(1,length(var_lst))*2;
   frc_case = [frc_case '_dummy_delete'];
   pro_year(2)=pro_year(1);
   Ncpu = 10;
   disp(['=================================================='])
   disp(['| dummy run. testing with minimum configuration. |'])
   disp(['=================================================='])
end

round_count=0;
tic
for i_var=1:length(var_lst)
for n_split=s_mem_split(i_var):N_mem_split(i_var)
    slice_time = tic;
    round_count=round_count+1;
    
    disp('----------------------------------------------------------------------------')
    disp(['Splitting the processing by ' num2str(N_mem_split(i_var)) ' to conserve memory!' ]);
    disp([' Round ' num2str(n_split) ' of ' num2str(N_mem_split(i_var))])
    disp('----------------------------------------------------------------------------')
    
    if Ncpu==0
        load_parpool=0;
    elseif round_count==1 
        load_parpool=1;
    elseif force_reload_parpool==0 
        if isempty(gcp('nocreate')) % parpool_loaded==0
            load_parpool=1;
        else
            load_parpool=0;
        end
    elseif force_reload_parpool~=0
        if round((round_count-1)/force_reload_parpool)==((round_count-1)/force_reload_parpool) && isempty(gcp('nocreate')) %parpool_loaded==0
            load_parpool=1;
        else
            load_parpool=0;
        end
    end
   
    
    % initialize velocity rotation
    uv_split=0;
    clear u_vel v_vel
    %for n_src=src_lst  
    
    for n_trg=C2M(var_lst(i_var))
        
        crop_raw = 1; %0: no cropping 1: cropping by predefined bounds 2: automatic cropping
        % automatic cropping does not yet work
        test_crop=0;
        
        circumpolar_model=0;
        store_field = 0; % stores the trg field after interpolation
        grab_stored_field=0;
        save2nc = 'on';
        
        %% transform dimensions
        perm2D = [2,1];
        perm3D = [2,1,3];
        perm4D = [2,1,3,4];
        perm5D = [2,1,3,4,5];
        
        %------------------------------------------------------------------
        %% assign processing year
        pro_year = [cpl_P2S_next-cpl_cycl+1  cpl_P2S_next];
        
        % for some fields let's request some more should time
        if any([30 32 33 34]==n_trg)&&n_src==25
           pro_year=[pro_year(1)-1 pro_year(2)+1]; 
        end
        
        %% assigning first raw nc file
        disp(['1) set parameter for data source ' num2str(n_src) '/' num2str(n_trg)])

        % initialize some values               
        if ~exist('frc_config','var') || length(frc_config)<n_trg || isempty(frc_config(n_trg).src_var_nme)
            retr_tme = tic;
            % quickly attain trg_lat and trg_msk to help frc_config
            % cropping the ncread field to a minimum 
            tmp_trg_lat=ncread(grd_file,'lat_rho');
            tmp_trg_lon=ncread(grd_file,'lon_rho');
            tmp_trg_msk=ncread(grd_file,'mask_oce');
            tmp_trg_lat(tmp_trg_msk==0)=NaN;
            tmp_trg_lon(tmp_trg_msk==0)=NaN;
            minmax_lat = tmp_trg_lat;
            minmax_lon = tmp_trg_lon;
            clear tmp_trg_lat tmp_trg_msk
            frc_config(n_trg) = frc_data_config(n_src,n_trg,trg_dpyr,force_cluster_machine,minmax_lat,minmax_lon);
            
            disp(['retrieve forcing configuration took ' num2str(toc(retr_tme)) 's.'])
        end
        
        src_fle_func =  frc_config(n_trg).src_fle_func;
        src_grd_func =  frc_config(n_trg).src_grd_func;
        src_pro_func =  frc_config(n_trg).src_pro_func;
        src_his_files = frc_config(n_trg).src_his_files;
        src_msk_func =  frc_config(n_trg).src_msk_func;
        src_tme_str=    frc_config(n_trg).src_tme_str;
        src_tme_unit=   frc_config(n_trg).src_tme_unit;
        src_tme_vec=    frc_config(n_trg).src_tme_vec;
        src_tme_ind=    frc_config(n_trg).src_tme_ind;       
        src_tme=        frc_config(n_trg).src_tme;
        src_dpyr=       frc_config(n_trg).src_dpyr;
        nc_var=         frc_config(n_trg).src_var_nme;
        tme_nme=        frc_config(n_trg).tme_nme;
        lat_nme=        frc_config(n_trg).lat_nme;
        lon_nme=        frc_config(n_trg).lon_nme;
        src_msk_nme=    frc_config(n_trg).src_msk_nme;
        xi_nme=         frc_config(n_trg).xi_nme;
        et_nme=         frc_config(n_trg).et_nme;
        met_coo=        frc_config(n_trg).met_coo;
        grd_typ=        frc_config(n_trg).grd_typ;
        trg_nc_var=     frc_config(n_trg).trg_nc_var;
        trg_tme_nme=    frc_config(n_trg).trg_tme_nme;
        trg_msk_nme=    frc_config(n_trg).trg_rmsk_nme;
        N_src_eta=      frc_config(n_trg).N_src_eta;
        N_src_xi=       frc_config(n_trg).N_src_xi;
        crp_ind=        frc_config(n_trg).crp_ind;
        knitting=       frc_config(n_trg).knitting;
        n_ex=           frc_config(n_trg).n_ex;
        fltr_w=         frc_config(n_trg).fltr_w;
        fld_cff=        frc_config(n_trg).fld_cff;
        scale_factor=   frc_config(n_trg).scale_factor;
        fld_offset=     frc_config(n_trg).fld_offset;
        msk_cfg=        frc_config(n_trg).mask_fill;
        fill_val=       frc_config(n_trg).fill_val;
        fill_NaN=       frc_config(n_trg).fill_NaN;
        NC_IND_STMP=    frc_config(n_trg).NC_IND_STMP; 
        NC_FLE_STMP=    frc_config(n_trg).NC_FLE_STMP;
        SRC2TRG_STMP=   frc_config(n_trg).SRC2TRG_STMP;
        SRC2TRG_TME =   frc_config(n_trg).SRC2TRG_TME;
        SRC2TRG_TME_VEC=frc_config(n_trg).SRC2TRG_TME_VEC;
        %N_trg_frm_ALL=  frc_config(n_trg).N_trg_frm;
        N_avg_cyc=      frc_config(n_trg).N_avg_cyc;
        N_rep_cyc=      frc_config(n_trg).N_rep_cyc;
        bld_trg_tme=    frc_config(n_trg).bld_trg_tme;
        trg_tme_str=    frc_config(n_trg).trg_tme_str;
        trg_tme_unit=   frc_config(n_trg).trg_tme_unit;
        trg_var_unit=   frc_config(n_trg).trg_var_unit;
        MAN_TRG_TME=    frc_config(n_trg).MAN_TRG_TME;
        xtra_fld_dim=   frc_config(n_trg).xtra_fld_dim;
        xtra_dim_sze=   frc_config(n_trg).xtra_dim_sze;
        src_3D      =   frc_config(n_trg).src_3D;
        src_prm_ind=    frc_config(n_trg).src_prm_ind;
        trg_prm_ind=    frc_config(n_trg).trg_prm_ind;
        apply_trg_msk=  frc_config(n_trg).apply_trg_msk;
        sub_case     =  frc_config(n_trg).sub_case;
        src_descr    =  frc_config(n_trg).src_descr;
        
        
        % create a local forcing file, copy it to target site later
        frcfile_final =  [pre_path sim_numbr '_frc_' sub_case frc_case '.nc'];
        if exist(frcfile_final,'file')
            delete(frcfile_final)
            disp(['deleted existing forcing file ' frcfile_final])
        end
        tmp_frc_cse = ['_tmp_' SOCS_SRC_CSE];
        frcfile = [pre_path sim_numbr '_frc_' sub_case tmp_frc_cse '.nc'];
        
        disp_txt = ['   | ' trg_nc_var ' from ' C2M(src_his_files(1)) ' src var: ' nc_var ' |'];
        disp(['   ' repmat('-',[1 length(disp_txt)-3])]);
        disp(disp_txt);
        disp(['   ' repmat('-',[1 length(disp_txt)-3])]);
        

        if n_split==s_mem_split(i_var) & i_var==1
%             tmp = pwd;
%             cd(prs_dir)
%             g = save_script([script_path '/fill_forcing_script_' sub_case frc_case],mfilename);
%             disp(['saved this script in .../fill_forcing_script_' sub_case frc_case ])
%             cd(tmp);
%             clear tmp
        end
        
        %% establish month and year index and costumize indexes
        % time will be read again, jointly with the field, here only month
        % and year indexing are done from the time vector provided by
        % calendar_index()
        
        src_mo_ind = SRC2TRG_TME_VEC(2,:)';
        src_yr_ind = SRC2TRG_TME_VEC(1,:)';
        
        src_mo_ind = reshape(src_mo_ind,size(SRC2TRG_TME));
        src_yr_ind = reshape(src_yr_ind,size(SRC2TRG_TME));
 
        disp(['Crop data series from ' num2str(minmax(src_yr_ind)) ' to ' num2str(pro_year)])
        
        trg_yr_ind = pro_year(1):pro_year(2);
        
        % remove individual entries
        del_ind = anyel(src_yr_ind,trg_yr_ind,1);
        
        NC_IND_STMP(del_ind~=1) = 0;
        NC_FLE_STMP(del_ind~=1) = 0;
        SRC2TRG_STMP(del_ind~=1) = NaN;
        
        % remove shelves
        del_ind = (sum(del_ind,1)==0);
        
        NC_IND_STMP(:,del_ind==1) = [];
        NC_FLE_STMP(:,del_ind==1) = [];
        SRC2TRG_STMP(:,del_ind==1) = [];
        SRC2TRG_STMP = SRC2TRG_STMP-minmin(SRC2TRG_STMP)+1;
        
        % reassign number of target frames
        N_trg_frm_ALL = size(SRC2TRG_STMP,2);
 
        %% create a sensible clustering of the processing
        if N_mem_split(i_var)>1 && N_trg_frm_ALL/N_mem_split(i_var)<2
            disp(['Possibly no need for memory splitting ? Check ! N_trg_frm: ' num2str(N_trg_frm_ALL) ' split by:' num2str(N_mem_split(i_var))]);
            return
        else
            % linearize vectors
            NC_IND_STMP = NC_IND_STMP(:);
            NC_FLE_STMP = NC_FLE_STMP(NC_IND_STMP~=0);
            SRC2TRG_STMP = SRC2TRG_STMP(NC_IND_STMP~=0);
            NC_IND_STMP = NC_IND_STMP(NC_IND_STMP~=0);
            
            % each position in vector corresponds to a source stamp, the
            % value at the position to the corresponding target frame:
            TRG_FRM_IND = [SRC2TRG_STMP(:)'];
            % find out how many sourc stamps are averaged for each target
            src_per_trg = ov(N_trg_frm_ALL);
            for n_frm=1:N_trg_frm_ALL
                src_per_trg(n_frm) = sumsum(TRG_FRM_IND==n_frm);
            end
            
            % here the splitting starts, on the metric of target frames
            pro_str = round(N_trg_frm_ALL/N_mem_split(i_var)*(n_split-1))+1;
            pro_end = round(N_trg_frm_ALL/N_mem_split(i_var)*n_split);
         
            pro_cnt = pro_end-pro_str+1;
            N_trg_frm = pro_cnt;
            
            % align splitting with target frames start and ends            
            ind_str = find(SRC2TRG_STMP==pro_str,1,'first');
            ind_end = find(SRC2TRG_STMP==pro_end,1,'last');
            
            NC_FLE_STMP = NC_FLE_STMP(ind_str:ind_end);
            SRC2TRG_STMP = SRC2TRG_STMP(ind_str:ind_end);
            NC_IND_STMP = NC_IND_STMP(ind_str:ind_end);
            
        end
        
        %% create gauss window for smoothing
        if exist('fltr_w','var') && any(fltr_w(:,1)>0)
            %H_win = gausswin(fltr_w) ./ sum(gausswin(fltr_w));
            for i_win=1:size(fltr_w,1)
                h_win = window2(fltr_w(i_win,1),fltr_w(i_win,1),@gausswin);
                h_win = h_win ./ sumsum(h_win);
                H_win(i_win) = {h_win};
            end
        end
        
        %% target grid coordinates, mask, reading from grid file
        disp(['2) establish target grid from ' grd_file])
        trg_latu=ncread(grd_file,'lat_u');
        trg_lonu=ncread(grd_file,'lon_u');
        trg_umask=ncread(grd_file,'mask_u');
        trg_latu = permute(trg_latu,perm2D);
        trg_lonu = permute(trg_lonu,perm2D);
        trg_umask = permute(trg_umask,perm2D);
        trg_umask(isnan(trg_umask)) = 1;
        trg_umask(trg_umask==0) = NaN;
        
        trg_latv=ncread(grd_file,'lat_v');
        trg_lonv=ncread(grd_file,'lon_v');
        trg_vmask=ncread(grd_file,'mask_v');
        trg_latv = permute(trg_latv,perm2D);
        trg_lonv = permute(trg_lonv,perm2D);
        trg_vmask = permute(trg_vmask,perm2D);
        trg_vmask(isnan(trg_vmask)) = 1;
        trg_vmask(trg_vmask==0) = NaN;
        
        trg_latr=ncread(grd_file,'lat_rho');
        trg_lonr=ncread(grd_file,'lon_rho');
        %trg_rmask=ncread(grdfile,'mask_rho');
        trg_rmask=ncread(grd_file,trg_msk_nme);
        trg_latr = permute(trg_latr,perm2D);
        trg_lonr = permute(trg_lonr,perm2D);
        trg_rmask = permute(trg_rmask,perm2D);
        trg_rmask(isnan(trg_rmask)) = 1;
        trg_rmask(trg_rmask==0) = NaN;
        trg_rang = ncread(grd_file,'angle');
        trg_rang = permute(trg_rang,perm2D);
        trg_pm = permute(ncread(grd_file,'pm'),perm2D);
        trg_pn = permute(ncread(grd_file,'pn'),perm2D);
        trg_srf_area = 1./trg_pn*1./trg_pm;
        % establish u angle and v angle (linear interpolation, good enough
        % the purpose here: rotation of u/v fields)
        trg_vang = (trg_rang(1:end-1,:)+trg_rang(2:end,:))/2;
        trg_uang = (trg_rang(:,1:end-1)+trg_rang(:,2:end))/2;
        trg_uang = fix_pi(trg_uang);
        trg_vang = fix_pi(trg_vang);
        
        
        trg_imask = ncread(grd_file,'zice');
        trg_imask = permute(trg_imask,perm2D);
        
        trg_imask(trg_imask~=0) = NaN; % if there is ice, value becomes NaN
        trg_imask(trg_imask==0) = 1;   % no ice value is one
        
        % only if two adjacent cells are ice shelf free the cell interface, i.e.
        % velocity point exists
        u_imask = trg_imask(:,2:end) .* trg_imask(:,1:end-1);
        v_imask = trg_imask(2:end,:) .* trg_imask(1:end-1,:);
        
        %% find min/max lat/lon of target domain
        lat_N = maxmax(trg_latr);
        lat_S = minmin(trg_latr);
        lon_W = minmin(trg_lonr);
        lon_E = maxmax(trg_lonr);
        
        %% assign appropriate target grid
        switch grd_typ
            case 'r'
                trg_lat = trg_latr;
                trg_lon = trg_lonr;
                % masks from a grid file:  1: valid point NaN: not valid
                %all_mask = trg_imask .* trg_rmask;
                all_mask = trg_rmask;
                trg_ang = trg_rang;
                
            case 'u'
                trg_lat = trg_latu;
                trg_lon = trg_lonu;
                all_mask = u_imask .* trg_umask;
                trg_ang = trg_uang;
            case 'v'
                trg_lat = trg_latv;
                trg_lon = trg_lonv;
                all_mask = v_imask .* trg_vmask;
                trg_ang = trg_vang;
        end
        if apply_trg_msk~=1
            all_mask(:)=1;
            disp('no target mask will be applied')
        end
        N_mskNaN = sumsum(isnan(all_mask));
        % linearized target coordinates of un masked points only.
        trg_lat_ = trg_lat(all_mask(:)==1);
        trg_lon_ = trg_lon(all_mask(:)==1);
        
        %% establish continuous angle at -2pi/2pi break of trg_rangd
        if circumpolar_model==1
            %circ_trg_ang = trg_ang;
            %circ_trg_ang(circ_trg_ang<0)=circ_trg_ang(circ_trg_ang<0)+2*pi;
            %circ_trg_ang_id = trg_rang<-3/4*pi|trg_rang>3/4*pi;
        end
        
        %% reading source coordinates and time of source maps (only need to do that once)
        src_grd_file = src_grd_func(1);
        src_msk_file = src_msk_func(1);
        src_file = src_fle_func(1);

        %% attempt to find out type of output
        if contains(src_file,'_avg_')
            tme_at_interval_end=1;
        else
            tme_at_interval_end=0;
        end
        
        disp(['   establish source grid from ' src_grd_file])
        if ~isempty(msk_cfg)
            disp(['   establish source mask from ' src_msk_file])
        end
        %if strcmpi(src_file(end-2:end),'.nc')
        SRC_LAT = ncread(src_grd_file,lat_nme);
        SRC_LON = ncread(src_grd_file,lon_nme);
        if ~isempty(src_prm_ind) && ndim(SRC_LAT)==2
            SRC_LAT = permute(SRC_LAT,[2 1]);
            SRC_LON = permute(SRC_LON,[2 1]);
        end
%            src_tme = ncread(src_file,tme_nme);
%         elseif strcmpi(src_file(end-2:end),'mat')
%             load(src_file, lat_nme, lon_nme, tme_nme);
%             eval(['SRC_LON = ' lon_nme ';']);
%             eval(['SRC_LAT = ' lat_nme ';']);
%             eval(['trg_sim_num = ' tme_nme ';' ]);
%             clear(lat_nme,lon_nme,tme_nme);
%         end        

        %% cropping coordinates to attain dimensions
        % assign croping bounds
        % will be overwritten with automatic bounds if crop_raw==2
        et_s=crp_ind(1); et_e=crp_ind(2); xi_s=crp_ind(3); xi_e=crp_ind(4);
        
        if crop_raw==0 %no cropping full array will be read
            if ndim(SRC_LAT)==2
                dim_et = size(SRC_LAT,1);
                dim_xi = size(SRC_LAT,2);
            elseif ndim(SRC_LAT)==1
                dim_et = length(SRC_LAT);
                dim_xi = length(SRC_LON);
            else
                disp('strange format of source coordinates')
            end
            
            et_s = 1;
            et_e = dim_et;
            xi_s = 1;
            xi_e = dim_xi;
            xi_c = dim_xi;
            et_c = dim_et;
        else % cropping
            dim_et = et_e-et_s+1;
            dim_xi = xi_e-xi_s+1;
            et_c = et_e-et_s+1;
            xi_c = xi_e-xi_s+1;
        end
        
        
        %% comparison plot of input domain and output domain
        if test_crop==1 && n_split==1
            % adapting longitude for continuous east coordinates
            SRC_LON(SRC_LON<=0) = SRC_LON(SRC_LON<=0) + 360;
            if ndim(SRC_LAT)==2
                domain_comp(trg_lonr,trg_latr,SRC_LON(et_s:et_e,xi_s:xi_e),SRC_LAT(et_s:et_e,xi_s:xi_e))
            else
                domain_comp(trg_lonr,trg_latr,SRC_LON(xi_s:xi_e),SRC_LAT(et_s:et_e))
            end
            % test / compare source and target grid resolution
            src_res_lat = latlon2metric_V3(SRC_LAT(:),repmat(SRC_LON(1),[numel(SRC_LAT(:)), 1]),'dist');
            src_res_lat = diff(src_res_lat);
            
            src_res_lon_t = latlon2metric_V3(repmat(SRC_LAT(1),[numel(SRC_LON(:)),1]),SRC_LON(:),'dist');
            src_res_lon_d = latlon2metric_V3(repmat(SRC_LAT(end),[numel(SRC_LON(:)),1]),SRC_LON(:),'dist');
            src_res_lon_m = latlon2metric_V3(repmat(SRC_LAT(round(length(SRC_LAT)/2)),[numel(SRC_LON(:)),1]),SRC_LON(:),'dist');
            src_res_lon = [diff(src_res_lon_t) diff(src_res_lon_d) diff(src_res_lon_m)];
            
            trg_res_eta = latlon2metric_V3(trg_latr(:,1),trg_lonr(:,1),'dist');
            trg_res_xi = latlon2metric_V3(trg_latr(1,:),trg_lonr(1,:),'dist');
            
            disp([' target grid resolution:  ' num2str(minmax([minmax(diff(trg_res_eta)); minmax(diff(trg_res_xi))])) ]);
            disp([' source grid res.   lat:  ' num2str(minmax(src_res_lat)) ]);
            disp([' source grid res.   lon:  ' num2str(minmax(abs(src_res_lon))) ]);
            return
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        disp(['3) read source data variable ' nc_var])
        read_source_time = tic;
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % start collecting raw data from  nc file
        %        for i_fls=1:N_fls
        %% establish source file
        SRC_TME=[];
        SRC_FLD=[];
        m_ind = [];
        %--------------------------------------------------------------
        %% read & crop source coordinates
        
        % load grid
        % read source coordinates
        if crop_raw~=0
            et_c = et_e-et_s+1;
            xi_c = xi_e-xi_s+1;
        else
            xi_c = dim_xi;
            et_c = dim_et;
        end
        if ndim(ncread(src_grd_file,lat_nme))==1
            src_lat = ncread(src_grd_file,lat_nme,[et_s],[et_c]);
            src_lon = ncread(src_grd_file,lon_nme,[xi_s],[xi_c]);
            if ~isempty(msk_cfg)
                if ndim(ncread(src_msk_file,src_msk_nme))==2
                    src_msk = squeeze(ncread(src_msk_file,src_msk_nme,[xi_s et_s],[xi_c et_c]));
                else
                    src_msk = squeeze(ncread(src_msk_file,src_msk_nme,[xi_s et_s 1],[xi_c et_c 1]));
                end
            end
        else
            src_lat = ncread(src_grd_file,lat_nme,[xi_s et_s],[xi_c et_c]);
            src_lon = ncread(src_grd_file,lon_nme,[xi_s et_s],[xi_c et_c]);
            if ~isempty(msk_cfg)
                if ndim(ncread(src_msk_file,src_msk_nme))==2
                    src_msk = ncread(src_msk_file,src_msk_nme,[xi_s et_s],[xi_c et_c]);
                elseif ndim(ncread(src_msk_file,src_msk_nme))==3
                    % this needs alteration, time changing mask needs to be
                    % read within time loop
                    src_msk = ncread(src_msk_file,src_msk_nme,[xi_s et_s 1],[xi_c et_c 1]);
                end
            end
        end
        if met_coo==1
            if ndim(ncread(src_grd_file,xi_nme))==1
                src_xi = ncread(src_grd_file,xi_nme,[xi_s],[xi_c]);
                src_et = ncread(src_grd_file,et_nme,[et_s],[et_c]);
                disp('   loaded 1D metric coordinates from source')
            else
                src_xi = ncread(src_grd_file,xi_nme,[xi_s et_s],[xi_c et_c]);
                src_et = ncread(src_grd_file,et_nme,[xi_s et_s],[xi_c et_c]);
                disp('   loaded 2D metric coordinates from source')
            end
            
        end

        if ~isempty(src_prm_ind)
            if ndim(src_lat)==2
                src_lat = permute(src_lat,[2 1]);
                src_lon = permute(src_lon,[2 1]);
            end
            if ~isempty(msk_cfg) && ndim(src_msk)==2
                src_msk = permute(src_msk,[2 1]);
            end
        end
        if any(src_lon(:)<0)
            disp(['   convert to continuous longitude 0 ... 360'])
            src_lon(src_lon<0) = src_lon(src_lon<0) + 360;
        end
        %--------------------------------------------------------------
        %% loading raw (full size) data
        % dummy read to attain hidden dimension
        if isempty(xtra_fld_dim)
            % assuming field dimension is [xi eta tme]
            if isempty(src_3D)
                tmp_fld = single(ncread(src_file,nc_var,[xi_s et_s 1],[xi_c et_c 1],[1 1 1]));
            else
                tmp_fld = single(ncread(src_file,nc_var,[xi_s et_s 1 1],[xi_c et_c 1 1],[1 1 1 1]));
            end
        elseif xtra_fld_dim==3
            % check if the range of the extra dimension is  provided
            if isempty(xtra_dim_sze)
                disp(' not sure how much of that extra dimension I am supposed to use')
            else
               dim3_str = xtra_dim_sze(1);
               dim3_cnt = xtra_dim_sze(2);
            end
            
            % if field dimension is bigger then the extra
            % dimensions will be hidden in the time dimension
            tmp_fld = single(ncread(src_file,nc_var,[xi_s et_s dim3_str 1],[xi_c et_c dim3_cnt 1],[1 1 1 1]));
            
        end
        % infer size
        sz_src_fld = size(tmp_fld);
        clear tmp_fld
        
        % initialize field for reading
        
        SRC_FLD = single(zeros([dim_et dim_xi sz_src_fld(xtra_fld_dim) N_trg_frm]));
        
        SRC_TME = zeros(1, N_trg_frm);
        
        disp(['   implicit averaging: establish mean for each target frame while reading.'])
        
        N_nc = length(NC_IND_STMP);
        i_nc=1;
        i_frm_mem_slc=1; % counter for target frames of this slice
        N_pro_src2trg = 0;
        
        while i_nc<=N_nc
            nc_info = ncinfo(src_file);
            [nc_c, nc_inc] = nc_monotonic(NC_IND_STMP(i_nc:end),NC_FLE_STMP(i_nc:end));
            % limit to reading maximum records at once
            %nc_c = min(nc_c,max_nc_read);
            
            
            
            
            src_file = src_fle_func(NC_FLE_STMP(i_nc));
            disp(['   loading ' num2str(nc_c) ' stamps from ' src_file])
            
            % start time step
            nc_s = NC_IND_STMP(i_nc);
            
            % read field
            if isempty(xtra_fld_dim)
                if isempty(src_3D)
                    % assuming field dimension is [xi eta tme]
                    src_fld = single(ncread(src_file,nc_var,[xi_s et_s nc_s],[xi_c et_c nc_c],[1 1 nc_inc]));
                else
                    src_fld = squeeze(single(ncread(src_file,nc_var,[xi_s et_s src_3D nc_s],[xi_c et_c 1 nc_c],[1 1 1 nc_inc])));
                end
            elseif xtra_fld_dim==3
                % if field dimension is bigger then the extra
                % dimensions will be hidden in the time dimension
                src_fld = single(ncread(src_file,nc_var,[xi_s et_s dim3_str nc_s],[xi_c et_c dim3_cnt nc_c],[1 1 1 nc_inc]));
            end
            
            % read time, this was done previously already but we do it
            % again to maintain consistency with reading the field chunks
            src_tme = double(ncread(src_file,tme_nme,[nc_s],[nc_c],[nc_inc]));
            % make sure time vector is a column not a row
            if size(src_tme,1)>size(src_tme,2)
                if i_nc==1
                    disp([ ' permute src_tme to row vector'])
                end
                src_tme = permute(src_tme,[1 2]);
            end
            
            % convert time to [d] for consistent handling / processing etc
            % not doing that
            % src_tme = time_unit(src_tme,'d',src_tme_unit,'true','true','quiet',src_dpyr);
            
            %% idiot check time stamps
            disp('   idiot check of time stamps after reading from netcdf')
            
%            [red_ind] = find_redundant(src_tme,'silent');
            if any(src_tme<0)
                disp('src_tme has negative stamps')
                return
            elseif any(src_tme==0)
                disp('src_tme has zero stamps')
                return
            elseif any(isnan(src_tme))
                disp('src_tme has NaN stamps')
                return
            elseif any(find_redundant(src_tme,'silent')~=0)
                disp('src_tme has redundant stamps')
                return
            else
                disp('   - src_tme has no NaN, zero, subzeros or redundant stamps')
            end
            
            if ~isempty(src_prm_ind)
                % permute if needed
                src_fld = single(permute(src_fld, src_prm_ind)); % output [lat lon x time]
            end
            
            
            
            %% implicit mean building, to save memory
            % initialize on first loop
            trg_frm_ind = SRC2TRG_STMP(i_nc:(i_nc+nc_c-1))';
            % disp(['------------ ' num2str(minmax(trg_frm_ind)) ' -- '])
            for n_trg_frm=min(trg_frm_ind):max(trg_frm_ind)
                if isempty(xtra_fld_dim)
                    SRC_FLD(:,:,i_frm_mem_slc)   = SRC_FLD(:,:,i_frm_mem_slc)   + sum(src_fld(:,:,trg_frm_ind==n_trg_frm),  ndim(src_fld))/src_per_trg(n_trg_frm);
                elseif xtra_fld_dim==3
                    SRC_FLD(:,:,:,i_frm_mem_slc) = SRC_FLD(:,:,:,i_frm_mem_slc) + sum(src_fld(:,:,:,trg_frm_ind==n_trg_frm),ndim(src_fld))/src_per_trg(n_trg_frm);
                else
                    disp('strange dimension')
                    return
                end
                % implicit averaging works consistent and independent of
                % the calendar type 
                % disp(['added ' num2str(sum(trg_frm_ind==n_trg_frm)) ' src frms to trg frm: ' num2str(i_frm_mem_slc) ' | value ' num2str(sum(src_tme(trg_frm_ind==n_trg_frm))/src_per_trg(n_trg_frm))])
                if tme_at_interval_end==1
                    SRC_TME(i_frm_mem_slc) = max([SRC_TME(i_frm_mem_slc) src_tme(trg_frm_ind==n_trg_frm)']);
                else
                    SRC_TME(i_frm_mem_slc) = SRC_TME(i_frm_mem_slc) + sum(src_tme(trg_frm_ind==n_trg_frm))/src_per_trg(n_trg_frm);
                end
                %% idiot check time stamps
                if n_trg_frm==min(trg_frm_ind)
                    disp('   idiot check of time stamps at assigning')
                end
                if SRC_TME(i_frm_mem_slc)<0
                    disp('src_tme has negative stamps after implicit averaging')
                    return
                elseif SRC_TME(i_frm_mem_slc)==0
                    disp('src_tme has zero stamps after implicit averaging')
                    return
                elseif isnan(SRC_TME(i_frm_mem_slc))
                    disp('src_tme has NaN stamps after implicit averaging')
                    return
                end
               % count the source frames that have been accumulated in the target frame
               N_pro_src2trg = N_pro_src2trg + sum(trg_frm_ind==n_trg_frm);
               % if all processed set counter to zero and advance target
               % frame index by one
               if src_per_trg(n_trg_frm)==N_pro_src2trg
                   i_frm_mem_slc=i_frm_mem_slc+1;
                   N_pro_src2trg = 0;
                   
               end
            end
            
            % advance loop counter
            i_nc = i_nc+nc_c;
            
        end
        
        
        %% reassign (latest) source mask, it changes with source files
        % this needs alteration, time changing mask needs to be
        % read within time loop

        src_msk_file = src_msk_func(NC_FLE_STMP(N_nc));
        if ~isempty(msk_cfg)
            disp(['   reading src mask: ' src_msk_nme ' from ' [src_msk_file]])
            if ndim(ncread(src_grd_file,lat_nme))==1
                if ndim(ncread(src_msk_file,src_msk_nme))==2
                    src_msk = squeeze(ncread(src_msk_file,src_msk_nme,[xi_s et_s],[xi_c et_c]));
                else
                    src_msk = squeeze(ncread(src_msk_file,src_msk_nme,[xi_s et_s 1],[xi_c et_c 1]));
                end
             else
                if ndim(ncread(src_msk_file,src_msk_nme))==2
                    src_msk = ncread(src_msk_file,src_msk_nme,[xi_s et_s],[xi_c et_c]);
                elseif ndim(ncread(src_msk_file,src_msk_nme))==3
                    src_msk = ncread(src_msk_file,src_msk_nme,[xi_s et_s 1],[xi_c et_c 1]);
                end
            end
            disp(['   check if field and mask size are consistent: '])
            if size(src_msk,1)==size(SRC_FLD,1) && size(src_msk,2)==size(SRC_FLD,2)
                disp([' ... Yup. All good.'])
            else
                disp([' ... Nah. Try to permute.'])
                src_msk = permute(src_msk,[2 1]);
                
                if ~(size(src_msk,1)==size(SRC_FLD,1) && size(src_msk,2)==size(SRC_FLD,2))
                    disp([' ... didn''t work. STOP'])
                    return
                else
                    disp(['   worked!'])
                end
            end
        end

        % -----------------------------------------------------------------
        
        disp(['    took ' num2str(round_dec(toc(read_source_time),1)) 'seconds.'])

        %% assign back to src_fld
        
        src_fld=SRC_FLD;
        src_tme=SRC_TME;
        
        clear SRC_FLD SRC_TME
        
       
        %% simple manipulation field with coefficient (scaling, presign, offset)
        src_fld = src_fld + single(fld_offset);
        src_fld = src_fld .* single(fld_cff);
        
        %% adapting longitude and field matrix for continuous eastward coordinates
        % London 0 .... London 360
        
        if knitting==1

            % backside of the world may need some knitting
            % rearranging chunks of field
            if find(src_lon == min(src_lon)) > find(src_lon == max(src_lon))
                disp(['   knitting at date line - rearrange field in EW direction']);
                src_fld = src_fld(:,[find(src_lon == min(src_lon)):size(src_lon,1) 1:find(src_lon == max(src_lon))],:);
                if ~isempty(msk_cfg)
                    src_msk = src_msk(:,[find(src_lon == min(src_lon)):size(src_lon,1) 1:find(src_lon == max(src_lon))]);
                end
                src_lon = src_lon([find(src_lon == min(src_lon)):size(src_lon,1) 1:find(src_lon == max(src_lon))]);
            end
        end
        
        disp(['4) modify source data values '])
        
        %% corrections in raw data
        if ~isempty(fill_val)
            tmp_fld=src_fld;
            
            
            if length(fill_val)>2
                src_fld(:)=fill_val(end);
                for i_fv=1:length(fill_val)-2
                   src_fld(tmp_fld==fill_val(i_fv))= single(fill_val(end-1));    
                end
                disp(['    set values==any(fill_val(1:end-2)) to fill_val(end-1) and all others to fill_val(end)'])
            else
                disp(['    apply fill values fill_val(1)->fill_val(2)'])
                src_fld(tmp_fld==fill_val(1))= single(fill_val(2));
            end
            
            clear tmp_fld
        end
        %if any([21 22 23 6]==n_src) && ~isempty(msk_cfg)
        if ~isempty(msk_cfg)
            tmp_msk = zeros(size(src_msk));
            for imsk=1:length(msk_cfg)-1
               tmp_msk(src_msk==msk_cfg(imsk))=1;
            end
            src_msk=tmp_msk;
            clear tmp_msk;
        end
        
        if ~isempty(msk_cfg)
            disp(['    apply source mask '])
            if isempty(xtra_fld_dim)
                src_fld(repmat(src_msk,[1 1 N_trg_frm])==0) = single(msk_cfg(end));
            elseif xtra_fld_dim==3
                src_fld(repmat(src_msk,[1 1 1 N_trg_frm])==0) = single(msk_cfg(end));
            end
        end
        switch n_src
            case 9
                %       src_fld(src_fld>100|src_fld<-5)=NaN;
        end
        %% extrapolate
        % extrapolate raw data into NaN or zero spaces to compensate for
        % masking differences between target and source grid
        if n_ex>0
            disp(['    extrapolate source data on source grid by ' num2str(n_ex)])
            if load_parpool==1 && isempty(gcp('nocreate')) %parpool_loaded==0
                pool_timer=tic;
                disp('Initializing parallel pool ....')
                if isempty(gcp('nocreate'))
                    if force_cluster_machine==4
                        pc = parcluster('local');
                        pc.JobStorageLocation = getenv('TMPDIR');
                        parpool(pc, str2num(getenv('SLURM_CPUS_PER_TASK')),'IdleTimeout', 120);
                    else
                        pc = parcluster('local');
                        parpool(pc,Ncpu,'IdleTimeout', 120);
                    end
                end
                parpool_loaded=1;
                disp(['                          .... took ' num2str(toc(pool_timer)) 's'])
            end
            
            extrap_tme = tic;
            tmp_N = size(src_fld,3);
            
            parfor k =1:tmp_N
                src_fld(:,:,k) = single(nan_extrap(src_fld(:,:,k),n_ex,'h','noplot'));
            end
            disp(['    extrapolating took took ' num2str(round_dec(toc(extrap_tme),2)) 's'])
        end
        
        %% creates 2D coordinate matrices [dim_lat dim_lon]
        if ndim(src_lat)==1 % if dimensions are still 1dim
            [src_lon,src_lat] = meshgrid(src_lon,src_lat);
        end
        if met_coo==1 && ndim(src_xi)==1 % if dimensions are still 1dim
            [src_xi,src_et] = meshgrid(src_xi,src_et);
        end
        
        
        %% deal with rotation of u/v fields
        if any([3 4 5 6]==n_trg) % put in any source number of u/v fields
            % afterward both u and v component are defined on same grid
            % source coordinates are replaced
            
            if any([3 5]==n_trg)
                % grab u fld
                disp(['4c) save u field for rotation'])
                u_vel = src_fld;
                u_coo = cat(3,src_lat,src_lon);
            elseif any([4 6]==n_trg)
                % grab v fld
                disp(['4c) save v field for rotation'])
                v_vel = src_fld;
                v_coo = cat(3,src_lat,src_lon);
            end
            if exist('u_vel','var')&&exist('v_vel','var')
                % swap the complex velocity field for the original field
                disp(['4c) combine u/v fields in complex vector'])
                disp(['    swap coordinates, reassign src dimensions '])
                % field now sits on rho coordinates
                [src_fld,src_lat,src_lon] = kinetic_energy_V6(u_vel,v_vel,'vdir',u_coo,v_coo,'   ');
                dim_et = size(src_lat,1);
                dim_xi = size(src_lat,2);
                uv_split=1;
                disp('4d) source u/v field ready for rotation.')
                
            else
                uv_split=0;
            end
        end
        
        %% interpolate
        disp(['5) interpolate source to target grid '])
        interpolate_tme = tic;
        if any(fltr_w(:,1)>0)
            disp(['    smooth field with gaussian window of ' num2str(fltr_w(:)')])
        end
        
        if ~isempty(xtra_fld_dim)
            N_trg_frm = N_trg_frm*sz_src_fld(xtra_fld_dim);
            disp(['    extend # of frames to be interpolated by hidden dimension / factor ' num2str(sz_src_fld(xtra_fld_dim))])
            % here is the last time we need N_trg_frm
        end
        
        if any([3 4 5 6]==n_trg) && uv_split==0
            disp(['   first velocity component, no interpolation yet'])
            trg_fld = single(NaN([size(all_mask),N_trg_frm]));
            field_ready_for_fill=0;
        else
            if grab_stored_field==1
                load([pre_path 'pre_data/tmp_' trg_nc_var num2str(N_trg_frm) '_' num2str(n_split) '_storage.mat']);
            else
                disp(['   -- frames: ' num2str(N_trg_frm) ' || mem ' num2str(n_split) '/' num2str(N_mem_split(i_var)) '  ----------------'])

                if load_parpool==1 && isempty(gcp('nocreate')) %parpool_loaded==0
                    pool_timer=tic;
                    disp('Initializing parallel pool ....')
                    %delete(gcp('nocreate'))
                    if force_cluster_machine==4
                        pc = parcluster('local');
                        pc.JobStorageLocation = getenv('TMPDIR');
                        parpool(pc, str2num(getenv('SLURM_CPUS_PER_TASK')),'IdleTimeout', 120);
                    else
                        pc = parcluster('local');
                        parpool(pc,Ncpu,'IdleTimeout', 120);
                    end
                    parpool_loaded=1;
                    disp(['                          .... took ' num2str(toc(pool_timer)) 's'])
                end
                
                if any(fltr_w(:,1)>0)
                    %disp(['    smooth field with gaussian window of ' num2str(fltr_w)])
                    % experimental
                    for i_win=1:size(fltr_w,1)
                        h_win = C2M(H_win(i_win));
                        for i_frm=1:size(src_fld,3)
                            
                            fl_w = fltr_w(i_win,1); mi_l = fltr_w(i_win,2); ma_l = fltr_w(i_win,3);
                            % expand field in lon
                            tmp0_fld = src_fld(:,:,i_frm);
                            % this expansion only works on specific input
                            % fields
                            tmp1_fld = [tmp0_fld(:,end-fl_w+1:end) tmp0_fld tmp0_fld(:,1:fl_w)];
                            
                            smo1_fld = filter2(h_win,tmp1_fld);
                            smo0_fld = smo1_fld(:,fl_w+1:end-fl_w);
                            % discriminate lat
                            smo0_fld(src_lat<mi_l|src_lat>=ma_l) = tmp0_fld(src_lat<mi_l|src_lat>=ma_l);
                            % fill back to source field
                            %src_fld(fl_w:end-fl_w,fl_w:end-fl_w,i_frm) = smo0_fld(fl_w:end-fl_w,fl_w:end-fl_w);
                            src_fld(:,:,i_frm) = smo0_fld;
                        end
                    end
                end
                if met_coo==1
                    trg_fld = interp2_metric(src_et,src_xi,src_fld,trg_lat,trg_lon,all_mask,~(isnan(src_fld(:,:,1))),0);
                else
                    trg_fld = interp2_metric(src_lat,src_lon,src_fld,trg_lat,trg_lon,all_mask,~(isnan(src_fld(:,:,1))),0);
                end
                
                if store_field==1
                    save([pre_path 'pre_data/tmp_' trg_nc_var num2str(N_trg_frm)  '_' num2str(n_split) '_storage.mat'],'trg_fld','-v7.3');
                end
            end
            field_ready_for_fill=1;
        end
        if force_reload_parpool~=0 && ~isempty(gcp('nocreate'))  %parpool_loaded==1
            if force_reload_parpool==1 || round(round_count/force_reload_parpool)==(round_count/force_reload_parpool)
                % shut down parpool
                disp(['circumnavigate memory leak: shut down parallel pool and reload at the beginning of next loop.'])
                delete(pc.Jobs)
                delete(gcp('nocreate'))
                parpool_loaded=0;
            end
        end

        if exist('xtra_fld_dim','var') && ~isempty(xtra_fld_dim)
            N_trg_frm = N_trg_frm/sz_src_fld(xtra_fld_dim);
            disp(['    reduce N_trg_frm by ' num2str(sz_src_fld(xtra_fld_dim))])
        end
        
        disp(['   interpolation/loading took ' num2str(round_dec(toc(interpolate_tme),2)) 's'])
        
        %% rotate and split u/v fields before writing
        if uv_split==1
            disp('6) rotation on of u/v field on target grid')
            
            % trg_ang has already been treated for continuity at -2pi/2pi
            % no need for circumpolar_model==1 special treatment
            
            % I dont know why we need to negate the rotation matrix, but
            % results of the southern ocean wind fields clearly require
            % this to happen
            trg_ang = -trg_ang;
            
            % rotate:  *complex(0,1) is rotation by pi/2, i.e. 45
            % degrees
            
            %fill_var = bsxfun(@times,fill_var,complex(0,1).^(2/pi*ang_var));
            %fill_var = bsxfun(@times,fill_var,complex(0,1).^(2/pi*(-trg_ang)));
            trg_fld = bsxfun(@times,trg_fld,complex(0,1).^(2/pi*(trg_ang)));
            
            if any([3 5]==n_trg)
                trg_fld = single(real(trg_fld));
            elseif any([4 6]==n_trg)
                trg_fld = single(imag(trg_fld));
            end
            uv_split=0;
        end
        
        %% check for NaN in trg_fld which finished processing
        
        % all_mask is assembled from the masks in grid file, should be NaN
        % for non valid points and 1 for valid points
        tmp2 = trg_fld; tmp2(isnan(repmat(all_mask,[1 1 size(trg_fld,3) ])))=1;
        check4NaN(tmp2,0,' after finish processing trg_fld');
        
        %% figure out target time stamp
        % here the conversion takes place of 
        %  reference time / time unit / (year length i.e. day per year -
        %  not yet implemented, needs coding in time_stamp_converter)
        if isempty(MAN_TRG_TME) && numel(bld_trg_tme)==1
            if bld_trg_tme==0 % adjust src_tme to new target reference time
                trg_tme = time_stamp_converter_V2(src_tme,[trg_tme_unit,src_tme_unit], ...
                    {trg_tme_str;src_tme_str},[src_dpyr trg_dpyr]);
            elseif bld_trg_tme==1 % original source time, keep reference date
                trg_tme = time_stamp_converter(src_tme,[trg_tme_unit,src_tme_unit], ...
                    {'true'; 'true'},{src_tme_str;src_tme_str},[src_dpyr trg_dpyr]);
            elseif bld_trg_tme==2 % source time cropped by full years since (1-1-0000,00:00:0)  
              %  trg_tme = time_stamp_converter(src_tme,[NaN,src_tme_unit], ...
              %      {'true'; 'true'},{trg_tme_str;src_tme_str},[src_dpyr trg_dpyr]);
            elseif bld_trg_tme==4 % source time cropped to its minimum year and then added offset by trg_tme_str
                trg_tme = time_stamp_converter(src_tme,[trg_tme_unit,src_tme_unit], ...
                    {'true'; 'true'},{trg_tme_str;src_tme_str},[src_dpyr trg_dpyr]);
            end
            
            % retrieve true time vector & check consistency between source
            % time and target time
            src_tme_vec = calendar_index(time_unit(src_tme,'d',src_tme_unit),src_dpyr,src_tme_str);
            trg_tme_vec = calendar_index(trg_tme,trg_dpyr,trg_tme_str);
            if bld_trg_tme==4
               % crop time vector to its beginning 
               tme_offset = datenum([minmin(trg_tme_vec(:,1)) 0 0]) - datenum(trg_tme_str);
               trg_tme = trg_tme - tme_offset;
               trg_tme_vec(:,1) = trg_tme_vec(:,1) - minmin(trg_tme_vec(:,1));
            elseif allel(src_tme_vec,trg_tme_vec,'ord')~=1
                disp('time stamp conversion failed ...')
                return
            end
            
        elseif numel(MAN_TRG_TME)>1
            % target time manually preset in the header
            trg_tme = MAN_TRG_TME;
        else
            disp('targe time stamp undefined - add manually after finishing')
        end
        
        %% convert time back to matlab reference [0 0 0 0 0 0]
        % to derive the accurate years contained in the time series
        %trg_tme = time_stamp_converter(trg_tme,[trg_tme_unit,trg_tme_unit], ...
        %    {'true'; 'true'},[0;trg_tme_str],trg_dpyr);
        
        %% figure out number of years covered by cycle
        % figure out the number of full years containing time stamps
        % e.g. t=[12 Dec 2010, 3 January 2011] is equivalent 2 data years
        %a = datevec(trg_tme);
        N_yrs_of_orig_cycle = numel(unique(trg_tme_vec(:,1)));%ceil(maxmax(trg_tme))-floor(minmin(trg_tme))+1;
        
        %         %% repeat cycle sourced from data (add N_rep_cyc cycles to the field)
        %         if N_rep_cyc>0
        %             % briefly convert to years for easier processing (from target time unit)
        %             trg_tme = time_unit(trg_tme,'y',trg_tme_unit,0,0,'quiet',daysperyear);
        %
        %             disp(['6) repeat forcing cycle ' num2str(N_rep_cyc) ' times'])
        %             trg_fld = repmat(trg_fld, [1 1 N_rep_cyc+1]);
        %             % add time to the beginning
        %
        %             % expand series of time stamps accordingly
        %             a = trg_tme;
        %             for i=1:N_rep_cyc
        %                 trg_tme = [a trg_tme+N_yrs_of_orig_cycle];
        %             end
        %
        %             % convert back from years to target time unit
        %             trg_tme = time_unit(trg_tme,trg_tme_unit,'y',0,0,'quiet',daysperyear);
        %             clear a tmp_trg_tme
        %         end
        
        
        %% create a mean year and add [add_mean_fld] repeated cycles of it to the beginning of time series
        if N_avg_cyc>0
            %             disp(['7) add ' num2str(N_avg_cyc) ' years of an average field to the beginning'])
            %             mean_fld = zeros([size(all_mask),12]);
            %
            %             for i_mo=1:12
            %                 mean_fld(:,:,i_mo) = nanmean(trg_fld(:,:,i_mo:12:end),3);
            %             end
            %             mean_fld = repmat(mean_fld,[1 1 N_avg_cyc]); % make it two mean years
            %             trg_fld = cat(3,mean_fld,trg_fld);
            %
            %             for i=1:N_rep_cyc
            %                 trg_tme = [trg_tme(1:12) trg_tme+daysperyear];
            %             end
        end
        
        %% process field if neede with already processed fields
        if n_src==25 && n_trg==31 % establish coefficient between ROMS and PISM melting
        
            disp('7) establish ratio between melting in ROMS and PISM')
            if ~exist(frcfile,'file')
                disp('   forcing file with pism_ishthk [30] and pism_mask_ish [33] does not exist')
                return
            end
            % unit convention: [m/yr], melt is >0 for ablation <0 for
            % accretion
            
            % read PISM time from the current file / convert to year (assuming it is in days )
            pism_time = double(ncread(frcfile,'pism_time'))/trg_dpyr;
            
            [tmp_lst]=data_case_V05(pre_path,{'pism_time'},{'pism_ishthk'},[tmp_frc_cse]);
            [pism_tme_vec,pism_tme_ind] = calendar_index(tmp_lst.src_tme,tmp_lst.src_dpyr,tmp_lst.src_ref_dte);
            
            %[socs_tme_vec,socs_tme_ind] = calendar_index(trg_tme,trg_dpyr,trg_tme_str);
            % assess time and select records within desired time frame
            
            int_str = datenum2([pro_year(1) 1 1 0 0 0],trg_dpyr);
            int_end = datenum2([pro_year(end) 12 31 24 0 0],trg_dpyr);
            
            int_pism = datenum(pism_tme_vec);
            a = find(int_pism<=int_str,1,'last');
            b = find(int_pism>=int_end,1,'first');
            
            if isempty(a)
                a=1;
            else
               % check if a record is two months after the interval start
               [c,d]=min(abs(int_pism-int_str));
                if c<60
                    a=d;
                end
            end
            if isempty(b)
                b=length(int_pism);
            else
                % check if a record is two months after the interval start
                [c,d]=min(abs(int_pism-int_end));
                if c<60
                    b=d;
                end
            end
            
            % read thickness from PISM
            pism_thck = permute(single(ncread(frcfile,'pism_ishthk')),perm3D);
            % read PISM ice shelf mask
            ishl_mask = permute(single(ncread(frcfile,'pism_mask_ish')),perm3D);
            
            % calculate PISM simulated melt rate from ice loss between first
            % and last time shot weighted by the mean of ice shelf mask [1 end]
            % unit [m/s]
            pism_melt = (mean(ishl_mask(:,:,[a b]),3).*diff(pism_thck(:,:,[b a]),1,3))./diff(pism_time([a b]));
            %pism_melt = -pism_melt; %convert ice loss (<0) to melt rate (>0)

            %-----------------------
            % duplicate field up to the number of entries in trg_tme 
            % thus assigning a constant melt coefficient 
            % expand time series to include time in the f
            % pism_melt = repmat(pism_melt,[1 1 length(trg_tme)]);
            
            
            % interpolate PISM thickness change onto ROMS melt time
            % stamps 
            % providing indexing of separate Antarctic regions
            if ~exist('RSSM_regions','var')
                disp(['   load region file ' pre_path 'pre_data/mapping_' grd_case])
                load([pre_path 'pre_data/mapping_' grd_case],'RSSM_regions')
            else
                disp('   region file already loaded')
            end
            disp('-- ice mass balance [Gt/yr] -----------------------')
            disp(['|       loss   net    period'])
            disp(['| PISM  ' num2str(round(sumsum((pism_melt.*double(pism_melt>0)).*trg_srf_area)*1E-9)) ...
                '   '  num2str(round(sumsum(pism_melt.*trg_srf_area)*1E-9)) ...
                '   [' datestr(pism_tme_vec(a,:)) ' -> ' datestr(pism_tme_vec(b,:)) ']'])
            disp(['| SOCS  ' num2str(round(nansumsum(mean((trg_fld.*double(trg_fld>0)),3).*trg_srf_area)*1E-9)) ...
                 '   ' num2str(round(nansumsum(mean(trg_fld,3).*trg_srf_area)*1E-9)) ...
                 '   [' num2str(pro_year(1):pro_year(end)) ']'])
            disp('---------------------------------------------------')
            
            mlt_cff_ver=1;
            if mlt_cff_ver==1
                
                mlt_cff = (sumsum((pism_melt.*double(pism_melt>0)).*trg_srf_area)*1E-9)/ ...
                    (nansumsum(mean((trg_fld.*double(trg_fld>0)),3).*trg_srf_area)*1E-9);
                disp(['- assign uniform coefficient - mass loss only: ' num2str(round_dec(mlt_cff,2))])
                tmp_fld = ones(size(trg_fld)).*mlt_cff;
            elseif mlt_cff_ver==2
                disp(['- assign regionally differentiated coefficient - mass loss only'])
                % populate all records of trg_fld with its mean
                tmp = mean(trg_fld,3);
                for j_trg=1:N_trg_frm
                    trg_fld(:,:,j_trg) = tmp;
                end
                reg_map = double(RSSM_regions(4).reg_ind);
                
                REG_IND = unique(reg_map);
                REG_IND(REG_IND==0)=[];
                tmp_fld = zeros(size(trg_fld));
                % calculate melt volume for each region
                for i=1:length(REG_IND)
                    i_reg=REG_IND(i);
                    %                                melt spots only m>0
                    SOCS_mlt = nansum(bsxfun(@times,mean(trg_fld,3).*(trg_fld>0),trg_srf_area.*double(reg_map==i_reg)),[1 2]);
                    %SOCS_mlt = nansum(bsxfun(@times,mean(trg_fld,3),trg_srf_area.*double(reg_map==i_reg)),[1 2]);
                    % thickening and thinning because the ice is dynamic in
                    % PISM and local thickening and thinning is not straightforward
                    % associated with freezing and melting
                    PISM_mlt = nansum(bsxfun(@times,pism_melt.*(pism_melt>0),trg_srf_area.*double(reg_map==i_reg)),[1 2]);
                    %PISM_mlt = nansum(bsxfun(@times,pism_melt,trg_srf_area.*double(reg_map==i_reg)),[1 2]);
                    
                    mlt_cff = PISM_mlt./SOCS_mlt;
                    mlt_cff(PISM_mlt<=0)=1;
                    mlt_cff(SOCS_mlt<=0)=1;
                    tmp_fld = tmp_fld + bsxfun(@times,repmat(double(reg_map==i_reg),[1 1 N_trg_frm]),mlt_cff);
                end
            end
            % scale the calculated coefficient against the previous
            % coefficient

%             
%             tmp = data_case_V05(pre_path,'ishm_time','ishm',['frc_' sub_case]);
%             a1 = tmp.src_tme;
%             if size(a1,1)==1
%                 a1=a1';
%                 % avoid confusing datenum2 in case horizontal entry is 6
%             end
%             prev_tme = datenum2(a1,tmp.src_dpyr)+datenum2(tmp.src_ref_dte,tmp.src_dpyr);
%             clear a1
%             
            
            % replace the original ROMS melt rate with the new coefficient
            trg_fld = tmp_fld;
            clear tmp_fld
            
            %tbd fiddle with coefficients / negative values in PISM and
            %huge coefficients.  those are mainly caused because SOCS/ACOSM
            %has fewer ice shelves that need to take the full melt load of
            %PISM melting
            % this may be resolved in part by using PISM topograpny for creating ACOSM bathy or adding sudo ish as -10 lids
            % in ACOSM
            % transpose the time stamps to be in the future, shifted by
            % cpl_cycle
      
            % bounds (datenum) of new time interval for which forcing is used
            int_str = datenum2([cpl_P2S_next+1 1 1 0 0 0],trg_dpyr);
            int_end = datenum2([cpl_P2S_next+cpl_cycl 12 31 24 0 0],trg_dpyr);
            
            % find index in the previous interval mlt frc file and read
            % field (15 days are subtracted to make sure we are searching
            % within the last interval file)
%             if cpl_P2S_next>2006
%                 ai=find(prev_tme<(int_str-15),1,'last');
%                 prev_fld = ncread(char(tmp.fle_nmes(tmp.fle_ind(ai))),'ishm',[1 1 tmp.nct_ind(ai)],[Inf Inf 1]);
%                 prev_fld = prev_fld';
%                 
%                 % multiply current coefficient with previous coefficient
%                 trg_fld = bsxfun(@times,trg_fld,prev_fld);
%                 clear tmp prev_fld prev_tme
%             end
            
            
            % establish datenum for the time stamps inbetween
            tmp_dte_num = linspace(int_str,int_end,N_trg_frm);
            % add half month at each end of the target time interval, we
            % can do this as the melt coefficient will be a constant forcing
            tmp_dte_num(1)=int_str-15;
            tmp_dte_num(end)=int_end+15;
            
            % shift to target time reference and assign to target time 
            disp('--- replace time vector ---')
            disp(datestr2(trg_tme,trg_dpyr))
            trg_tme = tmp_dte_num-datenum2(trg_tme_str,trg_dpyr);
            disp('--- with')
            disp(datestr2(trg_tme,trg_dpyr))
        end
        if n_src==21 && n_trg==19 % dew point conversion to humidity
            disp('7) convert dew point to humidity')
            tair_file = strrep(frcfile,'qai','tai');
            A = single(permute(ncread(tair_file,'Tair',[1 1 pro_str],[Inf Inf pro_cnt]),[2 1 3]));
            %A = A(:,:,1:size(trg_fld,3));
            trg_fld = src_pro_func(A,trg_fld);
            clear A
        end
        if any([21 24]==n_src) && n_trg==8 % ERA5/MRI sea ice concentration to SST restoring coefficient
            disp('7) convert sea ice concentration to a meaningful sst restoring coefficient')
            if n_src==21
                tami_fle = strrep(frcfile,'sst','tam_wfl');
            elseif n_src==24
                tami_fle = strrep(frcfile,'sst','mri_wfl');
            end
            % read time | by definition trg_tme is a row vector - ensure
            % consistency for interpolation
            tami_tme = ncread(tami_fle,'swf_time');%,[pro_str],[pro_cnt]);
            if size(tami_tme,1)>size(tami_tme,2); tami_tme=tami_tme'; end
            % determine reading index for tami_fle
            tami_s = find(tami_tme<trg_tme(1),1,'last');
            tami_e = find(tami_tme>trg_tme(end),1,'first');
            if isempty(tami_s); tami_s=1; end
            if isempty(tami_e); tami_e=length(tami_tme); end
            
            % read tamura bring onto [eta xi tme]
            tami = permute(ncread(tami_fle,'swflux',[1 1 tami_s],[Inf Inf tami_e-tami_s+1]),[2 1 3]);%,[1 1 pro_str],[Inf Inf pro_cnt]),[2 1 3]);
            tami_tme = tami_tme(tami_s:tami_e);
            % collector for interpolated tamura flux
            tami_int = NaN(size(trg_fld));
            
            % bring tme to front -> [tme eta xi]
            tami = permute(tami,[3 1 2]);
            tami_int = permute(tami_int,[3 1 2]);
            % interpolate tamura onto new time shots            
            parfor i=1:size(tami,3)
                tami_int(:,:,i) = interp1(tami_tme,squeeze(tami(:,:,i)),trg_tme);
            end
            
            % bring into trg_fld format [eta xi tme]
            tami_int = permute(tami_int,[2 3 1]);
            % establish coefficient that switches between exponential and
            % linear interpolation
            % restoring tempereature = [dqdsst*Tfr + (1-dqdsst*SST)] 
            % cff=0 => dqdsst = exp(sic) freezing and stable conditions
            % cff=1 => dqdsst = sic      melting conditions
            
            cff = (tanh((tami_int-3E-7)*2E6*pi)+1)/2;
            cff(isnan(cff))=0;
            % exp_sic: 1 sea ice 0: no sea ice
            exp_sic = src_pro_func(trg_fld);
            trg_fld = (cff.*trg_fld+(1-cff).*exp_sic);
        end
        if n_src==21 && n_trg==1 % ERA5 srf heat fluxes combine with Tamura ice fluxes
            % --> shf = sic*tamh + (1-sic)*(era5_sht+era5_lht)
            % this only works if time vars are on identical reference and
            % same unit
            
            disp('7) establish srf heat fluxes ERA5 + Tamura')
            
            e5sh_fle = strrep(frcfile,'shf','ssht');
            dqdt_fle = strrep(frcfile,'shf','sst');
            tamh_fle = strrep(frcfile,'shf','tam_hfl');
         
            % interpolate tamura onto new time shots
            % read time | by definition trg_tme is a row vector - ensure
            % consistency for interpolation
            tamh_tme = ncread(tamh_fle,'shf_time');
            if size(tamh_tme,1)>size(tamh_tme,2); tamh_tme=tamh_tme'; end
            % determine reading index for tami_fle
            tamh_s = find(tamh_tme<trg_tme(1),1,'last');
            tamh_e = find(tamh_tme>trg_tme(end),1,'first');
            if isempty(tamh_s); tamh_s=1; end
            if isempty(tamh_e); tamh_e=length(tamh_tme); end
            
            % read tamura bring onto [eta xi tme]
            tamh = permute(ncread(tamh_fle,'shflux',[1 1 tamh_s],[Inf Inf tamh_e-tamh_s+1]),[2 1 3]);
            tamh_tme = tamh_tme(tamh_s:tamh_e);
            % collector for interpolated tamura flux
            tamh_int = NaN(size(trg_fld));
            
            % bring tme to front -> [tme eta xi]
            tamh = permute(tamh,[3 1 2]);
            tamh_int = permute(tamh_int,[3 1 2]);
            
            parfor i=1:size(tamh,3)
                tamh_int(:,:,i) = interp1(tamh_tme,squeeze(tamh(:,:,i)),trg_tme);
            end
            % bring into trg_fld format [eta xi tme]
            tamh_int = permute(tamh_int,[2 3 1]);
            tamh_int(isnan(tamh_int))=0;
            
            dqdt = permute(ncread(dqdt_fle,'dQdSST',[1 1 pro_str],[Inf Inf pro_cnt]),[2 1 3]);
            e5sh = permute(ncread(e5sh_fle,'shflux',[1 1 pro_str],[Inf Inf pro_cnt]),[2 1 3]);
            
            trg_fld = dqdt.*tamh_int + (1-dqdt).*(e5sh+trg_fld);
        end
        if n_src==21 && n_trg==2 % ERA5 evaporation+precipitation combine with Tamura ice production rate
            % --> ssf = sic*tams + era5_eva + era5_precip
            % this only works if time vars are on identical reference and
            % same unit
            
            disp('7) establish srf fresh water fluxes ERA5 + Tamura')
            e5pr_fle = strrep(frcfile,'swf','tpc');
            dqdt_fle = strrep(frcfile,'swf','sst');
            tami_fle = strrep(frcfile,'swf','tam_wfl');
            
             % interpolate tamura onto new time shots
            % read time | by definition trg_tme is a row vector - ensure
            % consistency for interpolation
            tami_tme = ncread(tami_fle,'swf_time');
            if size(tami_tme,1)>size(tami_tme,2); tami_tme=tami_tme'; end
            % determine reading index for tami_fle
            tami_s = find(tami_tme<trg_tme(1),1,'last');
            tami_e = find(tami_tme>trg_tme(end),1,'first');
            if isempty(tami_s); tami_s=1; end
            if isempty(tami_e); tami_e=length(tami_tme); end
            
            % read tamura bring onto [eta xi tme]
            tami = permute(ncread(tami_fle,'swflux',[1 1 tami_s],[Inf Inf tami_e-tami_s+1]),[2 1 3]);%,[1 1 pro_str],[Inf Inf pro_cnt]),[2 1 3]);
            tami_tme = tami_tme(tami_s:tami_e);

            % collector for interpolated tamura flux
            tami_int = NaN(size(trg_fld));
            
            % bring tme to front -> [tme eta xi]
            tami = permute(tami,[3 1 2]);
            tami_int = permute(tami_int,[3 1 2]);
            
            parfor i=1:size(tami,3)
                   tami_int(:,:,i) = interp1(tami_tme,squeeze(tami(:,:,i)),trg_tme);
            end
            tami_int(isnan(tami_int))=0;
            % bring into trg_fld format [eta xi tme]
            tami_int = permute(tami_int,[2 3 1]);
            % retrieve sea ice concentration and precipitation
            dqdt = permute(ncread(dqdt_fle,'dQdSST',[1 1 pro_str],[Inf Inf pro_cnt]),[2 1 3]);
            e5pr = permute(ncread(e5pr_fle,'swflux',[1 1 pro_str],[Inf Inf pro_cnt]),[2 1 3]);
            
            trg_fld = dqdt.*tami_int + (1-dqdt).*(trg_fld + e5pr);
            
        end
        %% convert time from matlab reference [0 0 0 0 0 0] to trg_tme_str
        % to derive the accurate years contained in the time series
        %trg_tme = time_stamp_converter(trg_tme,[trg_tme_unit,trg_tme_unit], ...
        %    {'true'; 'true'},[trg_tme_str;0],trg_dpyr);
        
        %% idiot check time stamps
        disp('   idiot check of time stamps just before writing out')
        if size(trg_tme,1)>size(trg_tme,2)
            tmp = trg_tme;
        else
            tmp = trg_tme';
        end
        if any(trg_tme<0)
            disp('trg_tme has negative stamps')
            return
        elseif any(isnan(trg_tme))
            disp('trg_tme has NaN stamps')
            return
        elseif any(find_redundant(tmp,'silent')~=0)
            disp('trg_tme has redundant stamps')
            return
        else
            disp('   - trg_tme has no NaN, zero, subzeros or redundant stamps')
        end
        clear tmp
        %% fill in for NaN in trg frg_fld if required
        if ~isempty(fill_NaN)
            disp(['8) all NaN in target field filled with ' num2str(fill_NaN)])
            trg_fld(isnan(trg_fld))=fill_NaN;
        end
   
        if strcmpi(save2nc, 'on') && field_ready_for_fill==1
            writenc_tme=tic;
            frc_vars = {trg_nc_var,N_trg_frm_ALL};
            if dfl_lvl>0
                create_ROMS_frc(frcfile,size(trg_rmask,2),size(trg_rmask,1),frc_vars,'single',dfl_lvl);
            else
                create_ROMS_frc(frcfile,size(trg_rmask,2),size(trg_rmask,1),frc_vars,'single');
            end
            
            
            %% separate hidden dimension
            if exist('xtra_fld_dim','var') && ~isempty(xtra_fld_dim)
                disp(['    explicit hidden dimension'])
                trg_fld = reshape(trg_fld,[size(all_mask) sz_src_fld(xtra_fld_dim) size(trg_fld,3)/prod(sz_src_fld(xtra_fld_dim))]);
            end
            %% permute var for filling
            if ~isempty(trg_prm_ind)
                trg_fld = permute(trg_fld,trg_prm_ind);
            end
            
            %% write to nc file
            disp('9) filling forcing nc file')
            if exist('xtra_fld_dim','var') && isempty(xtra_fld_dim)
                ncwrite(frcfile,trg_nc_var,trg_fld,[1 1 pro_str]); %,[1 1 pro_cnt]);
                disp(['Saved ' num2str(size(trg_fld,3)) ' records (start:' num2str(pro_str) ') to ' frcfile])
            elseif exist('xtra_fld_dim','var') && xtra_fld_dim==3
                ncwrite(frcfile,trg_nc_var,trg_fld,[1 1 1 pro_str]);%,[1 1 1 pro_cnt]);
                disp(['Saved ' num2str(size(trg_fld,4)) ' records (start:' num2str(pro_str) ') to ' frcfile])
            end
            %% write time to file
            ncwrite(frcfile,trg_tme_nme,trg_tme,[pro_str]);
            %% write attributes
            if n_split==1
                % ROMS checks this variable, populated bounds of 0 and
                % 1000years
                ncwrite(frcfile,'time',[0 1000*365.25]); % 0 to 1000 years

                ncwrite(frcfile,'lat_rho',trg_latr');
                ncwrite(frcfile,'lon_rho',trg_lonr');
                ncwrite(frcfile,'lat_u',trg_latu');
                ncwrite(frcfile,'lon_u',trg_lonu');
                ncwrite(frcfile,'lat_v',trg_latv');
                ncwrite(frcfile,'lon_v',trg_lonv');
                if strcmpi(trg_tme_unit,'d'); wrt_unt = 'days'; 
                elseif strcmpi(trg_tme_unit,'s'); wrt_unt = 'seconds';
                elseif strcmpi(trg_tme_unit,'m'); wrt_unt = 'months';
                elseif strcmpi(trg_tme_unit,'y'); wrt_unt = 'years'; end
                ncwriteatt(frcfile,'time','units',[wrt_unt ' since ' datestr(trg_tme_str,'yyyy-mm-dd HH:MM:ss')]);
                ncwriteatt(frcfile,trg_tme_nme,'units',[wrt_unt ' since ' datestr(trg_tme_str,'yyyy-mm-dd HH:MM:ss')]);
                ncwriteatt(frcfile,trg_nc_var,'source',[src_descr]);
                ncwriteatt(frcfile,trg_nc_var,'units',trg_var_unit);
                if ~isempty(scale_factor)
                    ncwriteatt(frcfile,trg_nc_var,'scale_factor',scale_factor);
                end
                if trg_dpyr==365; cal_unit='noleap';
                elseif trg_dpyr>365; cal_unit='gregorian';
                elseif trg_dpyr==360; cal_unit='360_day';end
                ncwriteatt(frcfile,trg_tme_nme,'calendar',cal_unit);
            end
            disp(['Saved ' num2str(numel(trg_tme)) ' time records (start ind:' num2str(pro_str) ') to ' frcfile])
            %clear trg_fld trg_tme
            field_ready_for_fill=0;
            disp(['   final permuting & filling nc file took ' num2str(round_dec(toc(writenc_tme),2)) 's'])
        else
            disp(['NO FIELD saved to ' frcfile])
            %clear trg_fld
        end
    end
    
    disp(['memory slice took :' num2str(round_dec(toc(slice_time),2)) 's'])

end
end
% making sure file only exists if script ran successfully
movefile(frcfile,frcfile_final)
frcfile=frcfile_final;
disp(frcfile)
disp(['Script execution :' num2str(round_dec(toc(script_timer),2)) 's'])
end
