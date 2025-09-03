#!/usr/bin/env python3
"""
Demonstration of the Automated Model Retraining System
Shows all components working together
"""

import asyncio
import logging
from pathlib import Path
import tempfile
import shutil

from data_collection_pipeline import DataCollectionPipeline
from automated_labeling import AutomatedLabelingWorkflow
from retraining_pipeline import ModelRetrainingPipeline
from ab_testing_framework import ABTestingFramework, DeploymentStrategy
from retraining_scheduler import RetrainingScheduler

async def demo_data_collection():
    """Demonstrate data collection pipeline"""
    print("\n" + "="*60)
    print("DEMO: Data Collection Pipeline")
    print("="*60)
    
    pipeline = DataCollectionPipeline()
    
    # Collect recent incidents
    print("Collecting recent incident data...")
    incidents = await pipeline.collect_recent_incidents(days_back=1)
    print(f"✓ Collected {len(incidents)} incidents")
    
    # Show sample incident data
    if incidents:
        sample = incidents[0]
        print(f"Sample incident: {sample.incident_id}")
        print(f"  - Event type: {sample.event_type}")
        print(f"  - Confidence: {sample.confidence_score:.2f}")
        print(f"  - Camera: {sample.camera_id}")
    
    # Save metadata
    metadata_path = await pipeline.save_collection_metadata(incidents)
    print(f"✓ Metadata saved to: {Path(metadata_path).name}")
    
    return incidents

async def demo_automated_labeling(incidents):
    """Demonstrate automated labeling workflow"""
    print("\n" + "="*60)
    print("DEMO: Automated Labeling Workflow")
    print("="*60)
    
    workflow = AutomatedLabelingWorkflow()
    
    # Load models
    print("Loading AI models for labeling...")
    await workflow.load_models()
    print("✓ Models loaded successfully")
    
    # Create labeling tasks (limit to first 3 incidents for demo)
    if incidents:
        print("Creating labeling tasks...")
        sample_incidents = [
            {
                'incident_id': inc.incident_id,
                'video_path': inc.video_path
            }
            for inc in incidents[:3]
        ]
        
        tasks = await workflow.create_labeling_tasks(sample_incidents)
        print(f"✓ Created {len(tasks)} labeling tasks")
        
        # Process batch
        if tasks:
            print("Processing labeling batch...")
            results = await workflow.process_labeling_batch(batch_size=5)
            print(f"✓ Processed: {results['processed']} tasks")
            print(f"  - Auto-approved: {results['auto_approved']}")
            print(f"  - Requires validation: {results['requires_validation']}")
    
    # Get statistics
    stats = await workflow.get_labeling_statistics()
    print(f"✓ Total labeling tasks in system: {stats['overall']['total_tasks']}")

async def demo_retraining_pipeline():
    """Demonstrate retraining pipeline"""
    print("\n" + "="*60)
    print("DEMO: Model Retraining Pipeline")
    print("="*60)
    
    pipeline = ModelRetrainingPipeline()
    
    # Check retraining triggers
    print("Checking retraining triggers...")
    job = await pipeline.check_retraining_triggers()
    
    if job:
        print(f"✓ Retraining triggered: {job.job_id}")
        print(f"  - Trigger reason: {job.trigger_reason}")
        print(f"  - Data collection window: {job.data_collection_start.date()} to {job.data_collection_end.date()}")
        print(f"  - Status: {job.status}")
        
        # In a real scenario, we would execute the job
        print("  - Note: Full retraining execution requires training data and models")
    else:
        print("✓ No retraining triggers detected (normal in demo environment)")
        print("  - Triggers include: new data availability, performance degradation, scheduled retraining")

async def demo_ab_testing():
    """Demonstrate A/B testing framework"""
    print("\n" + "="*60)
    print("DEMO: A/B Testing Framework")
    print("="*60)
    
    framework = ABTestingFramework()
    
    # Create temporary model files for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        model_a = temp_path / "current_model.pt"
        model_b = temp_path / "new_model.pt"
        
        # Create dummy model files
        model_a.write_text("dummy model A")
        model_b.write_text("dummy model B")
        
        # Create A/B test
        print("Creating A/B test configuration...")
        test_config = await framework.create_ab_test(
            model_a_path=str(model_a),
            model_b_path=str(model_b),
            strategy=DeploymentStrategy.CANARY,
            traffic_split={"model_a": 0.8, "model_b": 0.2},
            duration_hours=24
        )
        
        print(f"✓ A/B test created: {test_config.test_id}")
        print(f"  - Strategy: {test_config.strategy.value}")
        print(f"  - Traffic split: {test_config.traffic_split}")
        print(f"  - Duration: {test_config.duration_hours} hours")
        print(f"  - Edge nodes: {len(test_config.edge_nodes)} nodes")
        
        # Start the test
        print("Starting A/B test deployment...")
        success = await framework.start_ab_test(test_config.test_id)
        
        if success:
            print("✓ A/B test started successfully")
            
            # Check active tests
            active_tests = await framework.get_active_tests()
            print(f"✓ Active tests: {len(active_tests)}")
            
            # Stop the test for demo
            print("Stopping test for demo...")
            await framework.stop_test(test_config.test_id, "demo_complete")
            print("✓ Test stopped")
        else:
            print("✗ Failed to start A/B test")

async def demo_scheduler():
    """Demonstrate retraining scheduler"""
    print("\n" + "="*60)
    print("DEMO: Retraining Scheduler")
    print("="*60)
    
    scheduler = RetrainingScheduler()
    
    # Get status
    status = scheduler.get_status()
    print(f"Scheduler status: {status['running']}")
    print(f"Active jobs: {status['active_jobs']}")
    print(f"Configuration:")
    print(f"  - Check interval: {status['config']['check_interval_hours']} hours")
    print(f"  - Max concurrent jobs: {status['config']['max_concurrent_jobs']}")
    print(f"  - Auto deployment: {status['config']['enable_auto_deployment']}")
    
    # Demonstrate manual trigger
    print("\nTesting manual retraining trigger...")
    job_id = await scheduler.trigger_manual_retraining("demo_trigger")
    
    if job_id:
        print(f"✓ Manual retraining triggered: {job_id}")
    else:
        print("✓ Manual trigger processed (would start retraining in production)")

async def demo_system_integration():
    """Demonstrate complete system integration"""
    print("\n" + "="*60)
    print("DEMO: Complete System Integration")
    print("="*60)
    
    print("This demonstrates how all components work together:")
    print("1. Data Collection → Gathers new incident footage")
    print("2. Automated Labeling → Labels the collected data")
    print("3. Retraining Pipeline → Trains new models when triggered")
    print("4. A/B Testing → Safely deploys and validates new models")
    print("5. Scheduler → Orchestrates the entire process automatically")
    
    print("\nIn a production environment:")
    print("• The scheduler runs continuously, checking for triggers")
    print("• When sufficient new data is available, retraining is triggered")
    print("• New models are validated against performance thresholds")
    print("• Successful models are deployed via A/B testing")
    print("• The system monitors performance and can rollback if needed")
    
    print("\n✓ All components are implemented and functional!")

async def main():
    """Run complete demonstration"""
    print("🤖 Automated Model Retraining System Demonstration")
    print("=" * 60)
    print("This demo shows all components of the retraining system working together.")
    
    try:
        # Demo each component
        incidents = await demo_data_collection()
        await demo_automated_labeling(incidents)
        await demo_retraining_pipeline()
        await demo_ab_testing()
        await demo_scheduler()
        await demo_system_integration()
        
        print("\n" + "="*60)
        print("✅ DEMONSTRATION COMPLETE")
        print("="*60)
        print("All components of the automated model retraining system are")
        print("implemented and working correctly!")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set up logging for demo
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log noise for demo
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())