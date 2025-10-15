# II- Physics-Informed Neural Networks : Solving Differential Equations with Neural Networks

[[II_1_SolvingPDEs]]

[[II_1_1_PDEsHistory]]
- PDEs history from Laplace
- But chaos and partial information

[[II_1_2_NumericalMethods]]
- Spectral methods
- Finite difference/finite element methods

[[II_1_2_GreyBoxModeling]]
- information is partial
- calibrated laws
- including measurements

[[II_2_PINNs]]

[[II_2_0_HistoricalContext]]
- 1990s: neural networks for solving PDEs
- 2017: Raissi et al. PINNs

[[II_2_1_NeuralNetworks]]
- Universal approximation theorem
- Neural network architecture
- Overparameterization
- Training neural networks

[[II_2_2_PINNsConcept]]
- Automatic differentiation
- Loss function: data + physics
- Advantages: mesh-free, flexible
- Limitations: training challenges, interpretability

[[II_2_3_ImplementationExampleUsingJax]]
- Jax library
- Example: solving a Poisson equation
- Code walkthrough

[[II_3_PINNsForContinuumMechanics]]

[[II_3_1_EquationsOfContinuumMechanics]]
- Conservation laws
- Constitutive relations
- Boundary and initial conditions

[[II_3_2_PINNsApplicationsInContinuumMechanics]]
- Parallel architecture
- Flexible implementation of boundary conditions

[[II_3_3_PINNsInverseProblems]]
- Inverse problems
- Constitutive Equation Gap Method
- Material parameter identification

# III- Separable PINNs for Faster Training

- Motivation: training time, scalability
- Overview of alternative architectures

[[III_1_SeparablePINNs]]

- Separable neural networks
- Low rank approximations

[[III_2_AdaptationToAnyGeometry]]
- Grid sampling constraint
- Geometry mapping

[[III_1_2_ComputationalBenefits]]
- Allen-Cahn equation
- Continuum mechanics examples 

# IV- Improving Training of PINNs
- Motivation: training challenges, convergence issues

[[IV_1_ClassicalNeuralNetworksTechniques]]
- Learning rate schedules
- Regularization
- Normalization
- Activation functions

[[IV_2_FourierFeatureEmbedding]]
- Concept
- Implementation details
- Adaption to Separable PINNs

[[IV_3_AdaptiveSamplingAndAttention]]
- Adaptive sampling
- Attention mechanisms

# V- PINNs for Inverse Problems in Continuum Mechanics

[[V_1_InverseProblemsAndMaterialTesting2.0]]
- Material Testing 2.0 concept
- Classical methods: FEMU, VFM
- Gap methods
- Limitations with noisy/partial data

[[V_2_PINNsAsCEGM]]
- PINNs as a particular case of CEGM
- Advantages over traditional methods

[[V_3_RobustnessToNoise]]
- Gaussian noise
- More realistic noise models

[[V_4_MissingData]]
- Missing data challenges
- Extrapolation capabilities

[[V_5_AdaptationToAnyGeometry]]
- Geometry mapping
- Free Boundary conditions

[[V_6_ComplexConstitutiveModels]]
- Orthotropic linear elasticity

[[V_7_StrainReconstruction]]
- Physics-based reconstruction
- Comparison with classical methods

# VI- Propagating Uncertainty with PINNs

[[VI_1_UncertaintyQuantification]]
- Aleatoric vs epistemic uncertainty
- Importance in material parameter identification
- Techniques for uncertainty quantification

[[VI_2_OverviewOfPINNsForUQ]]
- Bayesian PINNs
- Ensemble methods
- Dropout techniques
- Polynomial chaos expansions

[[VI_3_PINN-PCE]]
- Integration of PINNs with polynomial chaos expansions
- Implementation details
- Benefits for uncertainty quantification

[[VI_4_ApplicationsToPoissonEquation]]
- Application of PINN-PCE to Poisson equation
- Results and discussion

[[VI_5_ApplicationsToCompositePlate]]
- Application of PINN-PCE to composite plate
- Results and discussion