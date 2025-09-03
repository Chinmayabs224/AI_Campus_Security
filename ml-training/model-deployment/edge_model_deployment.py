"""
Edge Model Deployment System for AI Campus Security

This module provides secure model distribution to edge devices with versioning,
rollback capabilities, performance monitoring, and automated update scheduling.
"""

import asyncio
import logging
import json
import hashlib
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import aiohttp
import aiofiles
import yaml
import ssl
import certifi
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os


@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str
    model_path: str
    model_hash: str
    model_size: int
    created_at: datetime
    performance_metrics: Dict[str, float]
    deployment_status: str = "pending"  # pending, deploying, deployed, failed, rollback
    edge_devices: List[str] = None
    
    def __post_init__(self):
        if self.edge_devices is None:
            self.edge_devices = []


@dataclass
class EdgeDevice:
    """Edge device information"""
    device_id: str
    device_name: str
    ip_address: str
    port: int
    api_key: str
    current_model_version: Optional[str] = None
    last_heartbeat: Optional[datetime] = None
    status: str = "offline"  # online, offline, updating, error
    hardware_info: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.hardware_info is None:
            self.hardware_info = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}


@dataclass
class DeploymentJob:
    """Model deployment job"""
    job_id: str
    model_version_id: str
    target_devices: List[str]
    deployment_type: str  # full, incremental, rollback
    status: str = "pending"  # pending, in_progress, completed, failed, cancelled
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Dict[str, str] = None  # device_id -> status
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.progress is None:
            self.progress = {}


class EdgeModelDeploymentSystem:
    """
    Comprehensive edge model deployment system with security, versioning,
    and automated management capabilities
    """
    
    def __init__(self, config_path: str = "config/deployment_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Initialize paths
        self.deployment_dir = Path(self.config['deployment']['base_dir'])
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.deployment_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # Initialize database
        self.db_path = self.deployment_dir / "deployment.db"
        self._init_database()
        
        # Initialize encryption
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Device registry
        self.edge_devices: Dict[str, EdgeDevice] = {}
        
        # Background tasks
        self._background_tasks = set()
        
        # Load registered devices
        asyncio.create_task(self._load_registered_devices())
    
    def _load_config(self, config_path: str) -> Dict:
        """Load deployment configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                'deployment': {
                    'base_dir': '../../data/model_deployment',
                    'max_concurrent_deployments': 5,
                    'deployment_timeout_minutes': 30,
                    'heartbeat_interval_seconds': 60,
                    'performance_monitoring_interval_minutes': 15,
                    'auto_rollback_on_failure': True,
                    'require_device_authentication': True
                },
                'security': {
                    'use_tls': True,
                    'verify_model_signatures': True,
                    'encrypt_model_transfer': True,
                    'api_key_length': 32
                },
                'versioning': {
                    'max_versions_per_device': 3,
                    'cleanup_old_versions': True,
                    'version_retention_days': 30
                },
                'monitoring': {
                    'performance_degradation_threshold': 0.1,
                    'error_rate_threshold': 0.05,
                    'latency_threshold_ms': 1000,
                    'memory_usage_threshold': 0.8
                },
                'scheduling': {
                    'enable_auto_updates': True,
                    'update_window_start': "02:00",
                    'update_window_end': "04:00",
                    'batch_size': 3,
                    'rollout_strategy': "canary"  # canary, blue_green, rolling
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for deployment system"""
        logger = logging.getLogger('edge_model_deployment')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = self.deployment_dir / "deployment.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _init_database(self):
        """Initialize SQLite database for deployment tracking"""
        with sqlite3.connect(self.db_path) as conn:
            # Model versions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    model_hash TEXT NOT NULL,
                    model_size INTEGER,
                    created_at TIMESTAMP,
                    performance_metrics TEXT,
                    deployment_status TEXT DEFAULT 'pending'
                )
            ''')
            
            # Edge devices table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS edge_devices (
                    device_id TEXT PRIMARY KEY,
                    device_name TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    port INTEGER,
                    api_key TEXT NOT NULL,
                    current_model_version TEXT,
                    last_heartbeat TIMESTAMP,
                    status TEXT DEFAULT 'offline',
                    hardware_info TEXT,
                    performance_metrics TEXT,
                    FOREIGN KEY (current_model_version) REFERENCES model_versions (version_id)
                )
            ''')
            
            # Deployment jobs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS deployment_jobs (
                    job_id TEXT PRIMARY KEY,
                    model_version_id TEXT NOT NULL,
                    target_devices TEXT NOT NULL,
                    deployment_type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    progress TEXT,
                    error_message TEXT,
                    FOREIGN KEY (model_version_id) REFERENCES model_versions (version_id)
                )
            ''')
            
            # Performance monitoring table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    model_version_id TEXT,
                    timestamp TIMESTAMP,
                    inference_time_ms REAL,
                    memory_usage_mb REAL,
                    cpu_usage_percent REAL,
                    gpu_usage_percent REAL,
                    error_rate REAL,
                    throughput_fps REAL,
                    FOREIGN KEY (device_id) REFERENCES edge_devices (device_id),
                    FOREIGN KEY (model_version_id) REFERENCES model_versions (version_id)
                )
            ''')
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for secure model transfer"""
        key_file = self.deployment_dir / "encryption.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Secure the key file
            os.chmod(key_file, 0o600)
            return key
    
    async def _load_registered_devices(self):
        """Load registered edge devices from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM edge_devices')
            
            for row in cursor.fetchall():
                device = EdgeDevice(
                    device_id=row[0],
                    device_name=row[1],
                    ip_address=row[2],
                    port=row[3],
                    api_key=row[4],
                    current_model_version=row[5],
                    last_heartbeat=datetime.fromisoformat(row[6]) if row[6] else None,
                    status=row[7],
                    hardware_info=json.loads(row[8]) if row[8] else {},
                    performance_metrics=json.loads(row[9]) if row[9] else {}
                )
                
                self.edge_devices[device.device_id] = device
        
        self.logger.info(f"Loaded {len(self.edge_devices)} registered edge devices")
    
    async def register_edge_device(self, device_info: Dict[str, Any]) -> EdgeDevice:
        """Register a new edge device"""
        device_id = device_info['device_id']
        
        # Generate API key
        api_key = self._generate_api_key()
        
        device = EdgeDevice(
            device_id=device_id,
            device_name=device_info['device_name'],
            ip_address=device_info['ip_address'],
            port=device_info.get('port', 8080),
            api_key=api_key,
            hardware_info=device_info.get('hardware_info', {})
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO edge_devices 
                (device_id, device_name, ip_address, port, api_key, hardware_info)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                device.device_id,
                device.device_name,
                device.ip_address,
                device.port,
                device.api_key,
                json.dumps(device.hardware_info)
            ))
        
        self.edge_devices[device_id] = device
        self.logger.info(f"Registered edge device: {device_id}")
        
        return device
    
    def _generate_api_key(self) -> str:
        """Generate secure API key for device authentication"""
        key_length = self.config['security']['api_key_length']
        return base64.urlsafe_b64encode(os.urandom(key_length)).decode('utf-8')
    
    async def register_model_version(self, model_path: str, performance_metrics: Dict[str, float]) -> ModelVersion:
        """Register a new model version for deployment"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Calculate model hash
        model_hash = self._calculate_file_hash(model_path)
        model_size = model_path.stat().st_size
        
        # Generate version ID
        version_id = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_hash[:8]}"
        
        # Copy model to deployment directory
        deployment_model_path = self.models_dir / f"{version_id}.pt"
        shutil.copy2(model_path, deployment_model_path)
        
        # Create model version
        model_version = ModelVersion(
            version_id=version_id,
            model_path=str(deployment_model_path),
            model_hash=model_hash,
            model_size=model_size,
            created_at=datetime.now(),
            performance_metrics=performance_metrics
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO model_versions 
                (version_id, model_path, model_hash, model_size, created_at, performance_metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                model_version.version_id,
                model_version.model_path,
                model_version.model_hash,
                model_version.model_size,
                model_version.created_at.isoformat(),
                json.dumps(model_version.performance_metrics)
            ))
        
        self.logger.info(f"Registered model version: {version_id}")
        return model_version
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    async def create_deployment_job(self, model_version_id: str, target_devices: List[str], 
                                  deployment_type: str = "full") -> DeploymentJob:
        """Create a new deployment job"""
        job_id = self._generate_job_id()
        
        job = DeploymentJob(
            job_id=job_id,
            model_version_id=model_version_id,
            target_devices=target_devices,
            deployment_type=deployment_type,
            progress={device_id: "pending" for device_id in target_devices}
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO deployment_jobs 
                (job_id, model_version_id, target_devices, deployment_type, created_at, progress)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                job.job_id,
                job.model_version_id,
                json.dumps(job.target_devices),
                job.deployment_type,
                job.created_at.isoformat(),
                json.dumps(job.progress)
            ))
        
        self.logger.info(f"Created deployment job: {job_id}")
        return job
    
    def _generate_job_id(self) -> str:
        """Generate unique job ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        return f"deploy_{timestamp}_{random_suffix}"
    
    async def execute_deployment_job(self, job_id: str) -> bool:
        """Execute a deployment job"""
        job = await self._get_deployment_job(job_id)
        if not job:
            self.logger.error(f"Deployment job not found: {job_id}")
            return False
        
        self.logger.info(f"Starting deployment job: {job_id}")
        
        try:
            job.status = "in_progress"
            job.started_at = datetime.now()
            await self._update_deployment_job(job)
            
            # Get model version
            model_version = await self._get_model_version(job.model_version_id)
            if not model_version:
                raise Exception(f"Model version not found: {job.model_version_id}")
            
            # Execute deployment based on strategy
            if self.config['scheduling']['rollout_strategy'] == "canary":
                success = await self._execute_canary_deployment(job, model_version)
            elif self.config['scheduling']['rollout_strategy'] == "rolling":
                success = await self._execute_rolling_deployment(job, model_version)
            else:
                success = await self._execute_batch_deployment(job, model_version)
            
            if success:
                job.status = "completed"
                self.logger.info(f"Deployment job {job_id} completed successfully")
            else:
                job.status = "failed"
                self.logger.error(f"Deployment job {job_id} failed")
            
            job.completed_at = datetime.now()
            await self._update_deployment_job(job)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Deployment job {job_id} failed: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()
            await self._update_deployment_job(job)
            return False
    
    async def _execute_canary_deployment(self, job: DeploymentJob, model_version: ModelVersion) -> bool:
        """Execute canary deployment strategy"""
        self.logger.info("Executing canary deployment")
        
        # Deploy to first device (canary)
        canary_device = job.target_devices[0]
        canary_success = await self._deploy_to_device(canary_device, model_version, job)
        
        if not canary_success:
            self.logger.error("Canary deployment failed")
            return False
        
        # Monitor canary for performance
        self.logger.info("Monitoring canary deployment...")
        await asyncio.sleep(300)  # Wait 5 minutes
        
        canary_healthy = await self._check_device_health(canary_device)
        if not canary_healthy:
            self.logger.error("Canary deployment unhealthy, rolling back")
            await self._rollback_device(canary_device, job)
            return False
        
        # Deploy to remaining devices
        remaining_devices = job.target_devices[1:]
        batch_size = self.config['scheduling']['batch_size']
        
        for i in range(0, len(remaining_devices), batch_size):
            batch = remaining_devices[i:i + batch_size]
            batch_success = await self._deploy_batch(batch, model_version, job)
            
            if not batch_success:
                self.logger.error(f"Batch deployment failed for devices: {batch}")
                # Rollback all deployed devices
                await self._rollback_deployment(job)
                return False
        
        return True
    
    async def _execute_rolling_deployment(self, job: DeploymentJob, model_version: ModelVersion) -> bool:
        """Execute rolling deployment strategy"""
        self.logger.info("Executing rolling deployment")
        
        batch_size = self.config['scheduling']['batch_size']
        
        for i in range(0, len(job.target_devices), batch_size):
            batch = job.target_devices[i:i + batch_size]
            
            batch_success = await self._deploy_batch(batch, model_version, job)
            if not batch_success:
                self.logger.error(f"Rolling deployment failed for batch: {batch}")
                await self._rollback_deployment(job)
                return False
            
            # Wait between batches
            if i + batch_size < len(job.target_devices):
                await asyncio.sleep(60)  # Wait 1 minute between batches
        
        return True
    
    async def _execute_batch_deployment(self, job: DeploymentJob, model_version: ModelVersion) -> bool:
        """Execute batch deployment to all devices simultaneously"""
        self.logger.info("Executing batch deployment")
        
        # Deploy to all devices concurrently
        tasks = []
        for device_id in job.target_devices:
            task = asyncio.create_task(self._deploy_to_device(device_id, model_version, job))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        success_count = sum(1 for result in results if result is True)
        total_count = len(results)
        
        if success_count == total_count:
            self.logger.info(f"Batch deployment successful: {success_count}/{total_count}")
            return True
        else:
            self.logger.error(f"Batch deployment partial failure: {success_count}/{total_count}")
            return False
    
    async def _deploy_batch(self, device_batch: List[str], model_version: ModelVersion, job: DeploymentJob) -> bool:
        """Deploy model to a batch of devices"""
        tasks = []
        for device_id in device_batch:
            task = asyncio.create_task(self._deploy_to_device(device_id, model_version, job))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return all(result is True for result in results)    
    a
sync def _deploy_to_device(self, device_id: str, model_version: ModelVersion, job: DeploymentJob) -> bool:
        """Deploy model to a specific edge device"""
        device = self.edge_devices.get(device_id)
        if not device:
            self.logger.error(f"Device not found: {device_id}")
            job.progress[device_id] = "failed"
            return False
        
        try:
            self.logger.info(f"Deploying model {model_version.version_id} to device {device_id}")
            job.progress[device_id] = "deploying"
            await self._update_deployment_job(job)
            
            # Check device connectivity
            if not await self._check_device_connectivity(device):
                raise Except(main())uno.r    asynci":
_main__== "__ name_f __n()


ishutdowem.t_systt deploymen awaianup
    
    # Cleus}")
    {stattatus:nt seployme print(f"Did)
   job_yment_job.lotus(depyment_sta.get_deployment_systemawait deplous = s
    statyment statut deplo  
    # Ge  )
failed'}"cess else '' if sucuccessfult {'s"Deploymenprint(fd)
    job_it_job.deploymenment_job(te_deploysystem.execudeployment_ss = await succe   
 e deployment# Execut       
eduler()
 chted_update_st_automasystem.starployment_await de
    nitoring()nce_moerforma_psystem.startployment_wait de
    acesnd serviart backgrou 
    # St
   
    )type='full'deployment_        ,
'edge_002']_001', vices=['edgetarget_de    n_id,
    sion.versioodel_vern_id=model_versio        mb(
ment_jooyeplm.create_doyment_systeeplob = await d_jloymentjob
    deployment depte  Crea  #
     
    )
         }45
me_ms': ence_ti     'infer  78,
     all': 0.ec     'r      .82,
 : 0'precision'       ,
     85: 0.     'mAP50'    
   trics={ormance_me     perf
   t",ity.pv8n_securolos/y/model"../..path=     model_n(
   sioverister_model_system.reg deployment_awaiton =  model_versin
   iow model versegister a ne# R    
_info)
    ce2e(deviicr_edge_devistestem.regeployment_sy await d   _info)
evice1dge_device(dister_esystem.reg deployment_wait   
    a   }
    }
 32
     e_gb': 'storag         b': 8,
   _g     'memory    NX',
    Xavier A Jetsonu': 'NVIDI 'gp        {
   e_info': 'hardwar      
  ,': 8080       'port
 101',8.1..16': '192ip_address,
        'te'th Gampus Souname': 'Ca'device_       
 ge_002',e_id': 'ed     'devicnfo = {
   e2_i devic   
    }
 }
       ': 32
    e_gb     'storag  
     y_gb': 8,or  'mem          ier NX',
Jetson Xavu': 'NVIDIA     'gp      {
  info': dware_   'har8080,
     : ort'    'p
    00',168.1.1ess': '192.   'ip_addre',
     Gatampus North ame': 'Cice_n    'dev    e_001',
id': 'edgce_devi
        '1_info = {icees
    devge devicr edegiste# R    )
    
em(loymentSystelDepem = EdgeModsystoyment_em
    deplystnt sploymealize de Initi 
    #  "
 "ent system"ploym demodelof the edge age  us"Example""   ain():
  def mg
async and testinsagele u Examp)


#plete"shutdown comtem t sysloymene model depo("Edgnff.logger.i       sel      
   ns=True)
exceptiorn_s, retuground_taskck(*self._baherncio.gat   await asy      asks:
   und_trockgf self._ba i       lete
s to compit for task      # Wa
          el()
canc   task.     ks:
    ound_tasgr self._back infor task     asks
   ackground tncel bCa
        #      tem")
   syseployment edge model ddown "Shutting o(.infself.logger      "
   system""nt deploymeutdown   """Shlf):
     hutdown(sesync def s
    a  : {e}")
  d}ion_irson {veersip vd to cleanuf"Faile.error(.logger self               s e:
     Exception a    except           
                   }")
  rsion_idversion: {vep old model "Cleaned uinfo(f.logger.      self      
                           id,))
 ersion_ = ?', (vversion_idons WHERE odel_versi FROM mLETEe('DEn.execut    con        
        ebasatam d# Delete fro              
                        link()
  ath).unodel_p Path(m                       ts():
.exisodel_path)(math P       if          le
    fite model# Dele                   y:
      tr      ons:
     in old_versith  model_pasion_id,   for ver       
           ()
   tchall cursor.fe =nsiors   old_ve      
         ),))
      ormat(te.isoff_daof, (cut        '''L
    sion IS NUL_model_ver.current edND A < ?eated_atHERE mv.cr          Wion
      model_versed.current_= _id N mv.versiones ed Oe_devicIN edg     LEFT JO        mv
   s ionodel_vers   FROM m        h 
     v.model_paton_id, mCT mv.versi  SELE       '
       ute(''ecr = conn.ex    curso      deployed
  currently t nold versions    # Get o  
        conn:b_path) asct(self.dte3.conne sqliith      w     
    _days)
 etentionelta(days=row() - timedtetime.nate = daf_d   cutof
     n_days']retentio['version_']sioningverig['elf.conf = son_daysentiret"
        policy""n retention ed oions basrsl vemodelean up old "C""       lf):
 ions(sep_old_versef cleanu  async d   }
    
      e_info
 ar.hardwvice': dedware_info        'harrics,
    ormance_metrf': device.peance_metrics    'performe,
        se Noneartbeat elevice.last_hmat() if dforsoheartbeat.it_e.lasat': deviclast_heartbe         '
   el_version,urrent_modce.c deviion':del_versnt_mo'curre      us,
      ice.stats': devatu      'st
      _name,.deviceice': dev_nameice'dev   ,
         ce_idevice.devivice_id': d       'deurn {
     et  r        
     rn None
        retu:
     f not device   i    id)
 ice_get(devvices._de self.edge    device =   """
 e statusdge devic"Get e        ""ny]]:
ct[str, A[Dinaltio -> Opid: str)ice_evelf, dus(s_stat get_device def
    async    }
       age
 ror_messe': job.ererror_messag  '
           else None,leted_atob.comprmat() if jisofoeted_at. job.complpleted_at':  'com    
       None,elseat ted_ob.starat() if j.isoformarted_at': job.strted_atta   's
         (),rmatfoated_at.isot': job.created_a  'cre   
       gress,': job.proessprogr    '     us,
   : job.stats'tu'sta     id,
       _version_b.modelid': jorsion_ 'model_ve      _id,
     job.jobb_id':      'jo
       urn { ret
           
    eturn None       r     b:
t jo   if nob_id)
     joyment_job(deplof._get_await sel = ob j     
  us"""nt job statloyme""Get dep    ":
    ny]][str, Ational[Dict str) -> Op job_id:tus(self,ent_stadeploymet_ async def g     
 )
      )      evice_id
     device.d            ics),
mance_metre.perfor.dumps(devic    json      tus,
      evice.sta    d            
one,lse Nheartbeat et_lasvice. deat() if.isoformt_heartbeatvice.las de       
        n,l_versiode.current_mo device      
           ''', (          _id = ?
ERE device         WH       cs = ?
rmance_metrierfo    p           , 
     us = ?tat sat = ?,rtbe last_hean = ?,odel_versio_mT current SE     
          e_devices E edg     UPDAT           ''
te('nn.execu        co conn:
    ) asf.db_path(selctneqlite3.conwith s
        base"""n in datae informatiopdate devic    """U    Device):
ce: Edgef, deviin_db(selate_device_ef _upd   async d
    
 return None           
         )
            ow[6]
tatus=reployment_s       d            
 ] else {},row[5w[5]) if loads(ro=json.metricsrformance_   pe           ,
      row[4])misoformat(ime.froat=datetreated_      c           3],
   ize=row[el_s      mod       ,
       ow[2]l_hash=r       mode        
     [1],path=row      model_             w[0],
 ion_id=ro    vers           on(
     delVersi  return Mo       
         if row:       )
   .fetchone(sor row = cur            
         n_id,))
  versio''', (    ?
        rsion_id = RE veions WHEl_versM mode       FRO       
  nt_statusloymeics, depmance_metr   perfor                     
t,ed_aze, createl_simododel_hash,  model_path,sion_id, mSELECT ver           '
     cute('' conn.exersor =         cuconn:
   as _path) dbct(self.te3.conneli    with sq"
    abase""datsion from model ver"""Get        ion]:
 al[ModelVers) -> Option: str_idion verslf,n(seversiol_ef _get_mode    async d  ))
    

           job.job_id               sage,
.error_mes       job       s),
  progress(job.son.dump       j,
         e Noneat elscompleted_ job.format() ifeted_at.isoob.compl       j    ne,
     t else Noed_ajob.startt() if ormaofd_at.iste   job.star       ,
      .status      job          ''', (
       = ?
     RE job_id WHE           = ?
      _messageerroress = ?, , progrted_at = ? ?, compleat =started_us = ?,  SET stat        bs 
       loyment_joepATE d  UPD              
cute('''    conn.exe        as conn:
_path) elf.dbnect(sqlite3.conh s wit""
       se"b in databa joentloymepate d """Upd:
       loymentJob)lf, job: Depb(seyment_jopdate_deplonc def _u
    asye
    rn Non       retu   
  )
                  =row[9]
  sageror_mes      er       
       {},] else w[8f rorow[8]) iads(ess=json.lo progr            ,
       else Nonerow[7] (row[7]) if atfromisoformime.ted_at=datetmple    co                 None,
lserow[6] eif at(row[6]) romisoformatetime.f_at=ded start                   ow[5]),
t(romisoformatetime.fr_at=daedat     cre               ow[4],
tatus=r     s             [3],
  nt_type=rowdeployme             
       ),oads(row[2]son.lces=jt_devi     targe       ],
        ow[1ersion_id=r_vmodel                  
  0],b_id=row[          jo
          oymentJob(n Deplur ret            row:
             if   chone()
= cursor.fet     row        
    )
        , (job_id,)         '''d = ?
    job_is WHEREent_jobOM deploym          FR      r_message
 erroess,d_at, progr, complete, started_atted_at  crea                 atus,
    ent_type, stoymvices, depldetarget_d, _ionrsi model_veLECT job_id,       SE'
         n.execute(''sor = con       cur      conn:
as_path) dbonnect(self. sqlite3.c      with"
  "" databasejob fromdeployment t     """Ge]:
    ymentJobonal[Deplo -> Optiid: str)job_lf, (seyment_joblot_dep_gesync def     ahods
etlper matabase he    # Dbs
    
eturn jo 
        r      job)
 end(    jobs.app    
               )
         [5] else {} if rowads(row[5])on.los=js  progres                 (row[4]),
 at.fromisoformtetimeed_at=da      creat            =row[3],
  yment_type  deplo           
       ds(row[2]),oa.lvices=jsonrget_de       ta         
    id=row[1],ion_el_vers         mod          ],
 [0id=row  job_           
       entJob(= Deploym        job :
        etchall()or.fw in cursor ro f
                      
         ''')    reated_at
ER BY c      ORD          ing'
= 'pendE status ER          WHs 
      obnt_jployme FROM de           ss
    ogreated_at, pr cre                    _type, 
  deploymentces, vi, target_deersion_id_id, model_vECT job      SEL          '
n.execute(''or = concurs            conn:
 ) asdb_pathnect(self.ite3.conh sql wit     
          
= []      jobs 
   jobs"""eployment dngndi"Get pe""]:
        tJobment[Deploy> Liss(self) -t_jobmendeploypending_def _get_   async  
 
   (job.job_id)nt_jobe_deploymeself.execut   await              
id}")t: {job.job_enymduled deploing schecut.info(f"Exeoggerself.l              _jobs:
  endingin por job     f
                 obs()
   t_jng_deploymenet_pendiwait self._ging_jobs = a       pends
     oymentng deplr pendick fo      # Che     me:
 e <= end_tiimurrent_te <= c start_tim   ifndow
     update wiwithin ent time is if curr    # Check  
    
       ).time()"%H:%M"nd'], ate_window_eg']['updduling['schef.confiime(selrptst= datetime.time   end_me()
      ").ti "%H:%M'],art_stpdate_windowduling']['unfig['schef.co(selmestrptitime.me = date  start_ti    window
  se update # Par   
      ()
       me().tiatetime.nowe = drent_tim        cur"""
es model updatuledk for sched """Chec      s(self):
 update_scheduled_eck_forc def _chsyn   
    ascard)
 .diks_taskground_bacback(self.dd_done_callk.atas     
   (task)sks.addound_taackgr._blf  se())
      er_loop(scheduleate_tasksyncio.cr task = a   
             5 minutes
after # Retry ep(300) syncio.sle   await a             ")
    }: {eeduler errorche s"Updatror(fgger.erelf.lo           s        n as e:
 iopt Except exce     
          leep(3600)yncio.s as      await             ur
 ck every ho # Che                
                       _updates()
uledched_check_for_sait self.aw                  
      s']:_auto_updateable']['en['schedulingnfig if self.co                  
 y:tr            rue:
    e Twhil            p():
heduler_looync def sc    as    
    )
    cheduler"ate sated updoming autrtinfo("Stager.  self.log""
      "hedulere scpdatmated uauto"Start ""        (self):
hedulerscated_update__automartync def st    
    ass else 0
scoren_atio) if degradresation_scon max(degrad       retur   
    ation))
  , degradpend(max(0_scores.apionadat     degr
                      _val
      / baselinel)rrent_vane_val - culition = (baseradadeg            
        eors wower is     # L           
          else:          seline_val
 bal) /ne_vaaselit_val - b= (currenion egradat  d                 e
 s worsr iHighe  #                   te':
= 'error_ra metric =time_ms' orinference_== 'c  if metri        :
       _val > 0baseline if 
                     
  ric, 0)ent.get(meturrrent_val = c       cur
     0)metric, seline.get(ne_val = ba baseli        :
    key_metrics inicfor metr  
             ores = []
 n_sciodat       degra     
 ']
   ror_rates', 'erime_m_t'inferencetrics = [_me key       rics
 meton keycus   # Fo
      e"""ercentaggradation pe derformanculate pe"""Calc:
        ]) -> float floatt[str,urrent: Dic float], cr,ict[st baseline: Dn(self,egradatioce_dormanerfculate_p  def _cal    
  ne
rn No    retu  
             }
             0
 or] [5owut_fps': roughpthr       '             or 0,
 e': row[4]rror_rat        'e           or 0,
  3]ent': row[u_usage_perc'gp            0,
        w[2] or cent': roge_per'cpu_usa                  1] or 0,
  w[': rory_usage_mb   'memo                 r 0,
ow[0] oime_ms': rerence_tinf     '             return {
                w):
  x in roNone for  is not w and any(x   if ro       tchone()
  r.ferso cuw =   ro
                     )
id)_version_id, modelce_devi'', (    '   )
     ys'w', '-7 dadatetime('nomestamp > AND ti                ?
rsion_id = odel_ve m= ? ANDvice_id     WHERE de            ry 
toormance_hisOM perf          FR      ps)
t_fghpuAVG(throuror_rate), ent), AVG(erpercge_u_usa(gp        AVG             ent),
  e_percVG(cpu_usag Ae_mb),agusAVG(memory_e_time_ms), VG(inferencSELECT A                '''
e(onn.execut  cursor = c
          n:) as con_path.dbselfconnect( sqlite3.       with
        
 Nonen        retur:
     on_idel_versimodnot        if 
 """parisonr com metrics foancene perform"Get baseli  ""]:
      loat]ict[str, fal[Diontr]) -> Optptional[s Oon_id:odel_versi, mce_id: strlf, device(seanrmfoline_peret_base_gef c d  asyn 
     \n')
data) + 'ps(alert_.dumite(jsonait f.wr    awf:
        e, 'a') as ilrt_fes.open(aleofilsync with ai     ajsonl"
   lerts.e_amanc/ "perforir ent_df.deploym= selert_file       al  essing
procr later lert foStore a     #          
    }
  
    .isoformat()now()tetime.p': datimestam       's,
     ic': metricstr       'me    _type,
 ': alertlert_type 'a     
      ce_id,device.devi: ce_id'evi       'd = {
     rt_dataale      
  orsistratons to admintid notificaould senhis wl system, t a rea In      #    
      type}")
{alert_vice_id}: e.dedevicor device {ance alert f"Performarning(f.w self.logger
       """rmance alertfoger per """Trig  at]):
     flo, : Dict[strmetricsstr, alert_type: e, Devicdgef, device: Et(selrmance_alerrigger_perfoc def _t    asyn   
rics)
 rent_mettion', curce_degradamanice, 'perforce_alert(devmanrforgger_pet self._tri awai              :
 eshold']_thrdationgrae_deormancolds['perfn > threshegradatio if d         
  cs)_metricurrentne_metrics, li(baseradation_degrmanceulate_perfolcn = self._caiogradatde     
       trics:seline_me   if ban)
     el_versioent_modrrdevice.cuevice_id, ce(device.dperformanbaseline_f._get_selics = await etrbaseline_m      
  selinempared to ban codegradatioperformance  Check for  
        #
       metrics)current_usage', igh_memory_e, 'halert(devicmance_igger_perforf._trit sel        awa:
    threshold']ge_'memory_usaholds[, 0) > thresage'emory_usrics.get('mt_metif curren        age
ory us # Check mem    
       rics)
    ent_metency', curre, 'high_latt(devicormance_alerrigger_perflf._t sewait a          ld_ms']:
 hresho['latency_tdseshol 0) > thrtime_ms',ence_t('infert_metrics.geif curren      latency
  Check  #  
         s)
     t_metricte', currengh_error_raice, 'hi_alert(devmancer_perforigget self._tr      awai      hold']:
te_threserror_rasholds[' 0) > threte',_raror('erics.getnt_metrif curre      
  e ratck error   # Che        
   ring']
  nfig['monitoself.co= sholds hre     t"""
   sgger alertion and trigradatformance deck for per   """Che:
     tr, float])s: Dict[stric, current_meeDevicedevice: Edgself, ation(ce_degradrmank_perfodef _checsync  a
    
      ))        _fps', 0)
 ghputs.get('throu metric              , 0),
 r_rate''errot(.gemetrics                nt', 0),
ge_perceet('gpu_usa.g     metrics         ,
  cent', 0)_pert('cpu_usage  metrics.ge       0),
       ge_mb', 'memory_usaics.get(    metr            ', 0),
_time_mserenceet('inf metrics.g        
       t(),.isoformanow()   datetime.           d,
  ion_imodel_vers               _id,
 evice       d       
   ''', (          )
 ?, ?, ?,  ? ?, ?, ?, ?,ES (?, VALU           fps)
    throughput_or_rate, ent, err_perc_usageent, gpuge_perccpu_usa           
       usage_mb,emory_ime_ms, merence_testamp, infion_id, timl_versode mce_id,(devi              tory 
  ce_hisrman perfoT INTO       INSER        te('''
 execu conn.
           ) as conn:self.db_pathnect(3.contesqlith wi        base"""
ics in datance metrormarftore pe"""S    at]):
    str, floics: Dict[trl[str], meid: Optionasion_del_vertr, movice_id: self, de_metrics(sperformancestore_ _ync def 
    as
    {e}")device_id}: {device. device forileding faitorformance monror(f"Per.logger.er self
           : as eceptionpt Exexce       
           )
  in_db(devicedate_device__upself.wait           a)
  .now( = datetimeartbeatst_heice.la    dev  ics
       metrrics =ormance_met device.perf        trics
   nce meice performa Update dev  #      
              metrics)
  ion(device, ce_degradatk_performanself._chec    await       radation
  e degrmancperfo Check for          #       
   rics)
     sion, meterdel_vt_movice.currenid, device.device_metrics(deance_rme_perfo_storf. selit     awa
       databaserics in # Store met      
                n
  ur       ret      
   :trics meot     if n
       evice)ce_metrics(danice_perform._get_devit self awa metrics =       
    ent metrics# Get curr           
 y: tr       ice"""
ic devifspeca mance of tor perfor""Moni"        :
geDevice)device: Edce(self, erformance_ponitor_devic def _masyn   
 vice)
    e(de_performanctor_device self._moni     await          online":
 tatus == "evice.s   if d         ():
items_devices.self.edgeice in  device_id,for dev"
        ices""e of all devformanctor per"Moni""    ):
    vices(selfor_all_denit_mof  async de
    
   )cardks.disround_tas._backg(selfe_callbackadd_donask.)
        tadd(taskks.tasnd_ackgrou    self._b
    _loop())sk(monitortate_ncio.creask = asy
        ta
         minutefter 1 # Retry a) .sleep(60ioawait async                   }")
 error: {eng monitoriformance (f"Per.errorlf.logger    se                s e:
eption aexcept Exc                interval)
leep( asyncio.s      await        60
      nutes'] * ming_interval_rimance_monito']['perfordeploymentg['onfi.cl = selfnterva       i           
  ()_devicesor_alllf._monitawait se                  y:
          tr       :
  Truele   whi         or_loop():
c def monit     asyn 
          oring")
onitrformance m"Starting peogger.info(self.l      
  ring"""onitoance mperformackground "Start b      ""self):
  ng(orie_moniterformancef start_p  async d
    e)
  tions=Truxcepn_eturs, retaskllback_ro.gather(*asyncio     await ks:
       llback_tas   if ro  
           nd(task)
tasks.appeback_roll             )
   ob)ice_id, jvice(devllback_deelf._roreate_task(s.cyncio task = as               
d":"completeice_id) == get(devb.progress.   if jo         :
rget_devicesin job.taevice_id      for d= []
   _tasks llback       ro       
 
 job_id}") job {job.mentk deploying bacfo(f"Rollgger.in self.lo      "
 oyment""ntire deplck eRollba" ""     ntJob):
   Deploymelf, job:oyment(sellback_deplc def _ro
    asyn  e
  eturn Fals    r
        d}: {e}"){device_ievice r dk failed fobacrror(f"Roller.e self.logg         on as e:
  Excepti    except 
                       se
     al F  return             
         se: el                  success
   return                     
                             back"
 olled__id] = "rce[devib.progress jo                        vice)
   dece_in_db(date_devi._upt selfai  aw                       ion')
   s_verst('previou = result.gedel_version_moe.current  devic                       atus
   e device stat  # Upd                      
    success:       if                      
                   lse)
 , Fat('success'result.ge = ccess       su                
 .json()ait responselt = awresu                   200:
     atus == onse.st    if resp              
  nse:spoalse) as rers, ssl=Feadeaders=hhepost(url, sion.ith sesasync w           n:
     t) as sessioout=timeouimesion(tientSesClh aiohttp.sync wit         a=60)
   meout(totaltp.ClientTit = aiohtmeou ti              
              }
   son"
    ication/jpplype": "aent-T    "Cont            key}",
.api_er {devicef"Bearzation":    "Authori           {
      headers =      lback"
   /rol/api/modelsrt}.podevice}:{addressip_ice.://{devps= f"htt   url            
  )
        evice_id}" {ddeviceing back nfo(f"Rolllogger.ilf.      sery:
             t
 
        turn False    re
        evice:  if not d)
      et(device_iddevices.gf.edge_evice = sel     d
   ion"""ersdel vus moto previoice evback dRoll      """ bool:
  ntJob) ->ployme job: Dece_id: str,, devilfice(seck_devrollba async def _
   
    eturn None       re}")
     id}: {device_ce.ice {devics from devet metried to g"Failerror(fgger..lo      selfe:
      n as  Exceptioxcept      e        
              None
        return                  else:
              
         nse.json()respot eturn awai     r             200:
      .status == f response         i       nse:
    se) as resposl=Fals=headers, sader(url, he session.getsync with      a       
   ssion:meout) as seout=tision(timetp.ClientSeshtth aioasync wi       
     l=10)otantTimeout(tietp.Cloht aieout =       tim 
     
           ey}"}device.api_ker {Bearon": f""Authorizati = {ders        hea  rics"
  /met/apiice.port}{devip_address}:{device.f"https:// url =       y:
      tr      ""
 e"om devicrics france metent perform""Get curr
        "]:oat]flct[str,  Optional[Di->EdgeDevice) f, device: selmetrics(performance_ice_get_dev async def _   
   urn False
  ret       ")
    e_id}: {e}devicice { for devailedcheck fth f"Heal.error(elf.logger      se:
       as t Exception  excep 
                 True
return                
       n False
    retur           
   usage']}")['memory_: {metricsmory usagemehas high d} evice_if"Device {dng(.warnioggerself.l                ld']:
shoreage_thy_usors['memeshold, 0) > thremory_usage'cs.get('m if metri            
     
      eturn False         r      ]}ms")
 me_ms'e_ti'inferencmetrics[y: {gh latenchihas e_id} e {devicf"Devicg(ger.warnin   self.log            ']:
 ld_mseshoatency_thrhresholds['l0) > tme_ms', ti'inference_metrics.get(    if 
                    False
   return              e']}")
r_ratcs['erroe: {metrierror rathas high device_id} ce {Devi(f"er.warninggg self.lo     
          d']:_thresholerror_ratesholds[' 1.0) > threrate',.get('error_metrics if            
            oring']
'monitconfig[ self.thresholds =           sholds
 remance thperfor # Check            
       se
     Falurn     ret            
etrics:    if not m)
        cerics(deviance_metice_perform._get_devselfwait  metrics = a          e metrics
 performanct current   # Ge             try:
       
    rn False
       retu      device:
  not if     e_id)
   s.get(devicdge_devicelf.e se =  device
      t"""menr deploy health afteheck device"C""
        bool:: str) -> f, device_idealth(selk_device_h _checsync def
    a
    lseurn Faet           r
 {e}")ion: eptexcion odel activatror(f"Mr.er self.logge      e:
      as ionExceptxcept        e      
             
      urn False         ret         
      ext}")error_t.status} - {seled: {respon faiationtivl acf"Mode.error(ggerself.lo                       se.text()
  respon = awaitor_text    err               
      else:            
       se)alccess', F.get('sueturn result r                    
   son().jsponseawait re  result =                       == 200:
 tatusesponse.s     if r           
    as response:l=False) ders, ssders=heayload, hea=pal, jsonon.post(ur sessisync with         a      ssion:
 meout) as seut=tin(timeoClientSessioohttp.aic with asyn        
    0)tal=6(toentTimeoutCli= aiohttp.timeout         
              
    }
          rent': Truep_cur 'backu         id,
      rsion_l_version.vemodersion_id':         've {
        oad =     payl       
                   }

     tion/json"": "applicaeTyp"Content-      
          ey}",evice.api_kBearer {dation": f""Authoriz             s = {
   eader       hate"
     ivctels/a}/api/modce.portess}:{devie.ip_addrtps://{devic = f"ht   url     try:
         "
    device"" on edgeloyed modeleptivate d""Ac "       bool:
-> n)  ModelVersion:ioel_vers modce,EdgeDevice: e(self, devidel_on_devicivate_moc def _actsyn a
    
    False     return
       ")on: {e}er exceptidel transfr(f"Moger.erro self.log          ion as e:
 xcept Except 
        e                   
    return False                        )
rror_text}"- {ee.status} pons{res: nsfer failedtrar(f"Model er.erro  self.logg                    ext()
  se.twait responext = ar_t  erro                
             else:   
          e)ess', Falscc.get('suurn result         ret            son()
   esponse.j= await result       r                 == 200:
 status  response. if                   nse:
espolse) as rders, ssl=Fa headers=heapackage,n=model_ost(url, jsoion.pessith sync was      
          :ssionut) as semeout=timeosion(tientSes.Clihttpth aioc wi       asyns
     large modeltes for # 5 minual=300)  eout(totClientTim aiohttp. =     timeout        
                   }

    on"ication/jse": "applent-Typ"Cont          ,
      api_key}"evice.f"Bearer {d": zationAuthori "               rs = {
eade   h  "
       eploypi/models/dort}/a:{device.pddress}device.ip_ahttps://{f"url =       :
      
        try"""eviceedge de to odel packagTransfer m  """:
      ) -> bool, Any][strckage: Dicte, model_pae: EdgeDevic, devic(selfto_devicefer_model_c def _trans
    asyn   ckage
   return pa    
  }
        
        isoformat()().ime.nowetestamp': dat   'tim       metrics,
  erformance__version.p: modeletrics'ce_mperforman     ',
       'utf-8')ta).decode(ed_dancryptde(ee64.b64enco basa':'model_dat          '],
  transfercrypt_model_ty']['enrinfig['secu.coselfcrypted':    'en     _size,
    n.modelioel_versod_size': m    'model
        hash,odel_ersion.model_v': m'model_hash        n_id,
    on.versioel_versi_id': mod'version          = {
   kage
        pactadatage mee packa Creat       #
       
  _datadeled_data = morypt    enc             else:
_data)
   t(modeluite.encrypelf.cipher_s = sted_datacryp      en      r']:
odel_transfeypt_my']['encrig['securit self.conf   ifbled
      data if enaelypt modcr    # En    
    
    d()f.reaata = await     model_df:
        ) as h, 'rb'del_patiles.open(moofync with ai   asfile
     l  Read mode
        #
        ath)l_pversion.modeodel_ath(ml_path = P       mode"
 " transfer"forckage odel pa mptedrepare encry  """P:
       Any]ct[str,Diice) -> : EdgeDevn, devicersiolVeodesion: M, model_ver(self_packagemodelprepare_ _async def    
    alse
 F return     ")
      ce_id}: {e}{device.deviailed for check fonnectivity f"Device cger.warning(f.log      sel    
  as e:eption Excexcept            
                 = 200
nse.status =turn respo re                e:
   espons=False) as readers, sslers=h head.get(url,with session     async          sion:
  ut) as sesimeon(timeout=tlientSessioiohttp.Cth a    async wi      )
  t(total=10eouTimentttp.Cli aiohut =      timeo
             "}
     .api_key}{deviceearer ion": f"Borizats = {"Authder        health"
    t}/heavice.porress}:{dece.ip_addevis://{df"httprl =           u  y:
  tr"
      able"" reachise edge devicCheck if      """l:
   -> booDevice) gedevice: Ed, ivity(selfvice_connectheck_de_c async def    alse
    
turn F          re
  "ailed = "fdevice_id]ob.progress[    j
        : {e}")vice_id}{de device loy to depiled to(f"Farorf.logger.er  sel          tion as e:
cept Excep
        ex        rue
    eturn T    r    ")
    {device_id}ice del to dev moed deploySuccessfully(f"nfoogger.ielf.l    s      d"
  etempl] = "coss[device_idob.progre      j  
            ce)
    n_db(deviice_ie_devlf._updat await se      "
     "online= e.status     devic       ersion_id
 n.versio model_vel_version =rrent_modevice.cu d        s
   ice statu Update dev       #         
")
         failedivationl act"Modeon(xcepti Eise ra          cess:
     vation_suctit ac  if no       sion)
   del_verevice, mo_device(de_model_onivatf._actawait sel_success = onati  activ         
  on deviceodeltivate m    # Ac         
            failed")
ransferModel ton("aise Excepti     r
           s:nsfer_succes if not tra           l_package)
dece, moce(devidevir_model_to_._transfelfawait se= ess ccsu transfer_          device
 l to er mode    # Transf               
 ce)
    vi_version, deodelckage(mdel_pa_prepare_moelf. swait_package = a   modele
         kagodel pacypted mare encrPrep        # 
         ")
       ot reachablece_id} is nce {deviion(f"Devi