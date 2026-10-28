# Anàlisi: Cal fer Floquet per a estabilitat EMT amb models multilineals? 

Data: 2026-10-28

## Context
- Documentació del projecte a `deliverable/docs/` (proposta TenSyGrid, paper "Small-Signal Stability Analysis of Power Systems by Implicit Multilinear Models", etc.)
- Codi desenvolupat a `src/TensyGridEngine/` i `src/VeraGridEngine/`
- Punt actual: s'assumeix que cal fer Floquet per calcular estabilitat en models EMT, però no estem segurs. A més va molt lent.

## Respostes a preguntes
1. **Commutació**: Són models amb commutació (no promediats). Per sistemes amb commutació periòdica (PWM), el comportament és veritablement periòdic i l'anàlisi correcta és Floquet sobre el cicle límit. 
2. **Cobertura multilineal**: En principi hauria de cobrir tot el sistema (xarxa, càrregues, convertidors, màquines). Ara mateix la implementació sembla parcial/integració en curs.
3. **Objectiu**: Volem calcular l'estabilitat. El camí final segons TenSyGrid és passar a l'enfocament iMTI + valors propis generalitzats, però cal validar-lo contra Floquet.
4. **Velocitat**: El temps total és el problema. Conforme creixen el nombre de busos, el temps de càlcul es dispara (mala escalabilitat).
5. **Validació vs RMS**: Hi ha diferències entre RMS i EMT. Això justifica l'ús d'EMT per obtenir resultats correctes.
6. **Cas d'ús**: Necessitem això per a xarxes grans.

## Observacions inicials del codi

### Ruta actual (Floquet en domini temporal)
- `src/TensyGridEngine/small_signal_analysis_main_emt.py`: carrega xarxa EMT, força models multilineals (`FORCE_MULTILINEAR_MACHINE=True`), fa flux de càrrega, simula un període amb `EmtOptions` i executa `SmallSignalStabilityEmtDriver` amb `build_type=Arnoldi`/`HybridArnoldi`. Calcula multiplicadors de Floquet $\mu_i$ del monodromi $\Phi = \partial x(T)/\partial x(0)$, avalua estabilitat amb $|\mu_{max}|$ i guarda resultats.
- `src/VeraGridEngine/Simulations/SmallSignalStabilityEmt/small_signal_stability_emt_driver.py`: implementa `run_arnoldi()` (Arnoldi via SciPy sobre `EmtFloquetOperator`/`BlockEmtFloquetOperator` amb factoritzacions LU cachejades per pas) i `run_arnoldi_hybrid()` (BIRAM amb block-Arnoldi, deflació, reorthogonalització, restarts adaptatius). També suporta operador `AkStackBlockEmtFloquetOperator` i projecció tangent per DAE consistents (`ProjectedEmtFloquetOperator`).
- `src/VeraGridEngine/Simulations/SmallSignalStabilityEmt/emt_floquet_operator.py`: operador matricial lliure que propaga pertorbacions sobre tot el període integrant el sistema DAE EMT amb Jacobians/JIT; versió block per BLAS3; versió "Ak stack" explícita.
- Experiments amb reducció trigonomètrica: `emt/emt_ieee9_projected_trig_small_signal.py` projecta al manifold tangent de les relacions $\cos\theta,u_sin$-$\sin\theta,u_cos$; `emt/emt_ieee9_trig_manifold_reduced_small_signal.py` i `manifold_reduced_emt_floquet_operator.py` eliminen estats de lift trigonomètric per pas (redueixen dimensió). S'observen comparatives amb el cas "original machine".

### Ruta alternativa / relacionada amb multilinear
- `src/VeraGridEngine/Simulations/EMT/problems/emt_problem_multilinear.py`: afegeix utilitats per extreure matrius multilineals $(S,\Phi)$ (estructura monòmica) reutilitzant lògica de `RmsProblemMultilinear`. 
- `src/VeraGridEngine/Simulations/Rms/problems/rms_problem_multilinear.py`: infraestructura molt més extensa (CP/CPN, aproximadors, linearització eficient amb estructura tensorial). 
- Paper/proposta (TenSyGrid): proposen "small-signal stability analysis based on linearization of implicit multilinear models" calculant valors propis generalitzats a partir d'un model descriptor linealitzat de l'iMTI. També mostren guanys importants en temps de linearització amb estructura tensorial.
- `src/TensyGridEngine/fast_small_signal.py`: exemple de shift-and-invert sobre matrius $A,E$ precomputades (mètode pencil / per sistemes lineals/descriptors) - molt més ràpid conceptualment si ja tenim $A,E$.

## Raonament

### Quan cal Floquet vs valors propis generalitzats?
- **Sistemes LTI/linealitzats al voltant d'un punt d'equilibri** (RMS fasorial clàssic, o equilibri constant): l'estabilitat local ve donada per autovalors de $A$ (o valors propis generalitzats de $(A,B)$ / pencil). No cal Floquet.
- **Sistemes amb règim periòdic (limit cycle)**: si el sistema és periòdic en estat estacionari (commutació amb PWM/altres, o simplement operació periòdica), la linealització en sentit clàssic al voltant d'un punt no és vàlida; cal linealitzar al voltant de l'òrbita periòdica i obtenir els multiplicadors de Floquet de la matriu de monodromia. Estabilitat si tots $|\mu_i|<1$.
- **Models multilineals / trig-lift / transformacions**: el punt clau del treball TenSyGrid és que transformant amb models implícits multilineals (iMTI) i levantaments adequats (p.ex. representar $\sin/\cos$ o productes) es pot obtenir una formulació on la linealització dona un sistema lineal descriptor (temps-invariant en les coordenades transformades) i es poden calcular valors propis generalitzats directament, evitant la integració temporal sobre períodes per construir el monodromi. Si això és possible correctament, **no caldria Floquet** i seria molt més ràpid.

### Per què és lent ara?
- Cal simular fins a règim estacionari periòdic i després integrar **un període sencer** per **cada matvec** d'Arnoldi; amb $k_{krylov}$ iteracions i possiblement restarts, són molts períodes integrats. 
- Cada pas requereix resoldre sistemes DAE sparsos (LU per pas, cachejats). 
- Sistemes EMT grans (molts estats, xarxes amb convertidors) fan que cada resoldre sigui car. 
- Els experiments amb reducció trig (eliminar estats liftats) mostren que reduir dimensió ajuda, però cal assegurar que no es perd fidelitat.

### Punts forts / febles de cada enfocament

#### Enfocament A: Floquet (implementat actualment)
**Punts forts:**
- Teòricament correcte per a règim periòdic/veritable EMT amb commutació o dinàmica periòdica.
- General, funciona amb models existents sense reescriure tot com iMTI pur.
- Ja té infraestructura madura (operadors, block-Arnoldi, Ak-stack, reduccions, tangent consistent per DAE trapezoïdal).
- Permet analitzar directament el comportament sobre el cicle límit tal com està modelat.

**Punts febles:**
- Molt costós computacionalment (múltiples integracions de període + resolucions LU). 
- Depèn de convergència al cicle límit (ss_assessment_time), precisió del pas temporal (steps_per_period), etc.
- Difícil d'escalar a xarxes grans en temps raonable per a ús iteratiu.
- Menys "directe" que obtenir $A,E$ i fer pencil eigenvalue problem.

#### Enfocament B: Linearització directa de models multilineals (iMTI) amb valors propis generalitzats (proposat a TenSyGrid/paper)
**Punts forts:**
- Pot ser molt més ràpid si s'obté $(A,E)$ linealitzat (evita integració temporal repetida sobre períodes). Coherent amb el que diu el paper: "generalized eigenvalues are computed from linear descriptor models of the linearized implicit multilinear model".
- Compatible amb l'enfocament tensorial (CPN1) que dona guanys grans en linearització (ordre de magnituds segons literatura).
- Més similar al flux de treball clàssic (operating point -> linearize -> eigenvalues) familiar per a enginyers de xarxes.
- Escalable si estructura tensorial s'aprofita bé.

**Punts febles:**
- Requereix que el model estigui correctament formulat en forma multilineal/iMTI i que el levantament trigonomètric/productes estigui ben fet. Si hi ha commutació hard-switching o dinàmiques no capturades pel multilineal pur, cal veure validesa.
- Cal garantir que la linealització al voltant de l'equilibri de la formulació transformada correspon realment a la dinàmica local correcta (i que no s'han introduït artefactes amb els lifts).
- Infraestructura a mig desenvolupar comparada amb el camí Floquet ja integrat a VeraGridEngine.
- Necessari validar contra Floquet/time-domain amb cura (els experiments trig-manifold apunten en aquesta direcció).
