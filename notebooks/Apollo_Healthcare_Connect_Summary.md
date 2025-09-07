# Apollo Healthcare Connect: Multi-Modal Deep Learning Medical Triage System
## Summary and Key Insights

### Project Overview
**Author:** Glenn Dalbey  
**Institution:** Western Governors University - Data Science Program  
**Date:** August 10th, 2025  
**Live System:** apollohealthcareconnect.com  

### Research Question
Can a comprehensive multi-modal AI system be developed and deployed as a live production application to provide accurate healthcare routing between urgent care and emergency room facilities while maintaining clinical safety standards and providing comprehensive provider preparation materials?

## Key Problems Addressed

### Current Healthcare System Failures:
1. Patients with emergency conditions inappropriately book urgent care appointments
2. Extended waiting periods for time-sensitive conditions
3. Double payment burden (urgent care + ER transfers)
4. Urgent care facilities overwhelmed with out-of-scope patients
5. Healthcare providers unprepared for arriving patients

## Technical Architecture

### Multi-Modal System Components:
1. **Computer Vision Module:** Medical image analysis for burn and wound classification
2. **Natural Language Processing:** Symptom-based triage using DistilBERT
3. **Ensemble Architecture:** Five-model ensemble for robust predictions

### Model Components:
- **EfficientNet-B0, B1, B3:** Primary image classifiers
- **ResNet50:** Medical specialist model
- **DenseNet121:** Feature reuse for complex patterns
- **DistilBERT:** Text classification for symptom analysis

## Dataset Details

### Medical Image Data:
- **Total Images:** 8,085 across 8 medical conditions
- **Sources:** 4 Kaggle medical datasets
  - Burn images: 5,899 total
  - Wound images: 2,186 total
- **Class Imbalance:** Extreme 29.7:1 ratio

### Distribution:
- burn_1and2: 4,876 images (60.3%)
- burn_3rd: 1,023 images (12.7%)
- wound_pressure_wounds: 602 images (7.4%)
- wound_venous_wounds: 494 images (6.1%)
- wound_diabetic_wounds: 462 images (5.7%)
- Other wound types: <3% each

### Text Data:
- 250 balanced symptom descriptions
- Categories: "Urgent Care" vs "Emergency Room"
- Synthetic generation for realistic patient scenarios

## Performance Metrics

### Outstanding Results Achieved:
- **Burn Classification:** 98% accuracy (critical for 3rd degree burns)
- **Text Classification:** 94% accuracy for symptom triage
- **Combined Multi-Modal:** 93.8% overall accuracy
- **Image Classification:** 90.4% accuracy
- **Processing Speed:** 77 batches in 50 seconds (1,227 images)

### Technical Achievements:
1. Successfully handled 29.7:1 class imbalance ratio
2. Real-time inference capabilities (<1 second response)
3. Production deployment with AWS S3 and Render hosting
4. Confidence scoring for uncertainty quantification

## Advanced Techniques Implemented

### Class Imbalance Handling:
- **Focal Loss:** α=1, γ=2 for hard example focus
- **Label Smoothing:** 0.1 factor to prevent overconfidence
- **Balanced Oversampling:** 3,900 samples per class
- **Class Weights:** Computed using sklearn

### Data Augmentation Pipeline (15 steps):
- Geometric transformations (rotation, flipping)
- Optical/grid distortions
- CLAHE for medical image enhancement
- Noise simulation (Gaussian, ISO)
- CoarseDropout for occlusion handling

### Ensemble Strategy:
- Weighted voting across 5 models
- Conservative threshold (0.35) for medical safety
- Confidence analysis (mean: 0.556, std: 0.084)

## Clinical Impact

### Healthcare Workflow Improvements:
1. **Accurate Routing:** Prevents inappropriate urgent care bookings for emergency conditions
2. **Cost Reduction:** Eliminates double payment scenarios
3. **Provider Preparation:** Advanced patient information before appointments
4. **Resource Optimization:** Better allocation of healthcare facilities

### Safety Measures:
- Conservative routing for high-risk conditions
- Transparent confidence reporting
- Ensemble consensus for critical decisions

## Technical Infrastructure

### Production Deployment:
- **Frontend:** Flask web application
- **Storage:** AWS S3 for medical images
- **Hosting:** Render platform
- **CI/CD:** GitHub Actions
- **GPU:** NVIDIA RTX 3090 Ti (25.3GB)

### Processing Capabilities:
- 208.67 samples/second evaluation
- 19-second training for 75 steps
- Batch size: 16-64 depending on model

## Limitations Identified

1. **Dataset Constraints:**
   - Limited samples for rare conditions
   - Research-grade data vs. clinical validation
   - Geographic/demographic diversity lacking

2. **Computational Requirements:**
   - High-end GPU needed for ensemble
   - Linear scaling with model count
   - Real-time constraints for production

3. **Clinical Validation:**
   - No formal clinical trials conducted
   - Provider feedback not systematically collected
   - Long-term outcome tracking needed

## Future Research Directions

### Direction 1: Advanced Architecture
- Vision transformer integration
- Federated learning for multi-institutional training
- Automated ensemble weight learning
- Explainable AI components

### Direction 2: Healthcare Integration
- EHR system connectivity
- Population health analytics
- Predictive resource allocation
- Public health surveillance integration

## Key Innovations

1. **Multi-Modal Fusion:** Successfully combined CV and NLP for healthcare routing
2. **Production Deployment:** Live system serving real users worldwide
3. **Extreme Imbalance Handling:** Novel combination of focal loss and label smoothing
4. **Medical Safety Focus:** Conservative thresholds and confidence reporting

## Conclusions

The Apollo Healthcare Connect system demonstrates:
- **Technical Excellence:** 93.8% combined accuracy exceeds clinical thresholds
- **Practical Application:** Live deployment at apollohealthcareconnect.com
- **Healthcare Impact:** Addresses critical inefficiencies in patient routing
- **Scalability:** Cloud-based architecture ready for expansion

This project successfully bridges the gap between advanced AI research and practical healthcare applications, providing a foundation for continued innovation in medical AI systems.

---
*Summary created from the original Apollo Healthcare Connect capstone project report*
