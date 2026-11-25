"""Exact lifting, shared substitutions, and preservation of Block interfaces."""
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic import symbolic as s
from VeraGridEngine.Utils.Symbolic.symbolic_ml import multilinearize_block


def initialize(block, bindings):
    values = dict(bindings)
    pending = {v:expr for child in block.get_all_blocks() for v,expr in child.init_eqs.items() if v.uid not in values}
    while pending:
        ready = [(v,expr) for v,expr in pending.items() if all(x.uid in values for x in expr.get_vars())]
        assert ready, 'Cyclic or missing initialization dependencies'
        for var,expr in ready:
            values[var.uid] = expr.eval_uid(values)
            del pending[var]
    return values


def monomials(expr):
    """Independent multi-affinity checker: expand sets of variable factors."""
    if isinstance(expr,s.Const):
        return [set()]
    if isinstance(expr,s.Var):
        return [{expr.uid}]
    if isinstance(expr,s.UnOp):
        return monomials(expr.operand)
    assert isinstance(expr,s.BinOp), f'Non-polynomial runtime expression: {expr}'
    left,right=monomials(expr.left),monomials(expr.right)
    if expr.op in ('+','-'):
        return left+right
    if expr.op=='/':
        assert all(not factors for factors in right)
        return left
    assert expr.op=='*', f'Unconverted power: {expr}'
    result=[]
    for a in left:
        for b in right:
            assert not a&b, f'Repeated monomial variable in {expr}'
            result.append(a|b)
    return result


def assert_multilinear(block):
    for child in block.get_all_blocks():
        for eq in child.state_eqs+child.algebraic_eqs+child.differential_eqs:
            monomials(eq)


def test_repeated_trig_is_shared_across_hierarchy_and_separate_expression_nodes():
    vf=VarFactory();theta=vf.add_var('theta');dtheta=vf.add_diff_var('dtheta',base_var=theta)
    first,second=(vf.add_var(name) for name in ('first','second'))
    child=Block(algebraic_vars=[second],algebraic_eqs=[second-s.sin(theta)],out_vars=[second])
    block=Block(state_vars=[theta],state_eqs=[s.Const(2.)],diff_vars=[dtheta],
        algebraic_vars=[first],algebraic_eqs=[first-s.sin(theta)-s.sin(theta)*s.cos(theta)],
        children=[child],out_vars=[first],init_eqs={theta:s.Const(.3)})
    original=str(block.algebraic_eqs[0]);converted=multilinearize_block(block,vf)
    lifts=[v for b in converted.get_all_blocks() for v in b.state_vars if v.uid!=theta.uid]
    assert len(lifts)==2
    assert sorted(v.name for v in lifts)==['u_cos','u_sin']
    assert str(block.algebraic_eqs[0])==original
    assert len(block.children)==1
    assert converted.out_vars[0].uid==first.uid
    assert converted.children[0].out_vars[0].uid==second.uid
    values=initialize(converted,{theta.uid:.3,first.uid:math.sin(.3)*(1+math.cos(.3)),second.uid:math.sin(.3)})
    for b in converted.get_all_blocks():
        assert all(abs(e.eval_uid(values))<1e-12 for e in b.algebraic_eqs)
    assert_multilinear(converted)
    again=multilinearize_block(converted,vf)
    assert sum(len(b.state_vars)+len(b.algebraic_vars) for b in again.get_all_blocks()) == sum(
        len(b.state_vars)+len(b.algebraic_vars) for b in converted.get_all_blocks())


def test_equal_names_with_distinct_uids_are_not_shared():
    vf=VarFactory();a,b=(vf.add_var('theta') for _ in range(2))
    block=Block(state_vars=[a,b],state_eqs=[s.Const(1.),s.Const(2.)],
                diff_vars=[vf.add_diff_var('da',base_var=a),vf.add_diff_var('db',base_var=b)],
                algebraic_eqs=[s.sin(a)+s.sin(b)])
    converted=multilinearize_block(block,vf)
    assert sum(len(child.state_vars) for child in converted.get_all_blocks())==6
    names=[v.name for child in converted.get_all_blocks() for v in child.state_vars[0:] if v.uid not in (a.uid,b.uid)]
    assert len(set(names))==4


def test_existing_substitution_reuses_declared_variable():
    vf=VarFactory();theta,u,out=(vf.add_var(n) for n in ('theta','u_sin','out'))
    block=Block(in_vars=[theta,u],algebraic_vars=[out],algebraic_eqs=[out-s.sin(theta)-s.sin(theta)])
    result=multilinearize_block(block,vf,substitutions={s.sin(theta):u})
    assert not result.children
    assert result.algebraic_eqs[0].eval_uid({out.uid:1.,u.uid:.5})==0


@pytest.mark.parametrize('value',[-1.3,.2,2.1])
def test_polynomial_rational_root_and_piecewise_lifts_are_exact(value):
    vf=VarFactory();x=vf.add_var('x');y=vf.add_var('y')
    expr=(x+x*x)*(x-s.Const(.3)) + x**5 + 2/(x*x+2) + x**-2 + s.sqrt(x*x+1) + s.abs(x) + s.max(x,.1)
    source=Block(in_vars=[x],algebraic_vars=[y],algebraic_eqs=[y-expr])
    result=multilinearize_block(source,vf)
    values=initialize(result,{x.uid:value,y.uid:expr.eval_uid({x.uid:value})})
    assert_multilinear(result)
    for child in result.get_all_blocks():
        for eq in child.algebraic_eqs:
            assert abs(eq.eval_uid(values))<1e-9


def test_state_kinematics_match_original_nonlinear_trajectory():
    vf=VarFactory();x=vf.add_var('x');dx=vf.add_diff_var('dx',base_var=x)
    rhs=s.sin(x)+s.exp(-x)
    block=Block(state_vars=[x],diff_vars=[dx],state_eqs=[rhs],init_eqs={x:s.Const(.3)})
    converted=multilinearize_block(block,vf)
    assert_multilinear(converted)
    states=[v for b in converted.get_all_blocks() for v in b.state_vars]
    equations=[eq for b in converted.get_all_blocks() for eq in b.state_eqs]
    values=initialize(converted,{})
    algebraic=Block(init_eqs={v:eq for b in converted.get_all_blocks() for v,eq in b.init_eqs.items()
                             if v.uid not in {v.uid for v in states}})
    def lifted_rhs(t,point):
        bindings=initialize(algebraic,{v.uid:float(value) for v,value in zip(states,point)})
        return [eq.eval_uid(bindings) for eq in equations]
    times=np.linspace(0,1,101)
    lifted=solve_ivp(lifted_rhs,(0,1),[values[v.uid] for v in states],t_eval=times,rtol=1e-10,atol=1e-12)
    reference=solve_ivp(lambda t,y: np.sin(y)+np.exp(-y),(0,1),[.3],t_eval=times,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(lifted.y[0],reference.y[0],rtol=2e-9,atol=2e-10)
    for i,var in enumerate(states):
        if var.name=='u_sin': np.testing.assert_allclose(lifted.y[i],np.sin(lifted.y[0]),atol=2e-9)
        if var.name=='u_cos': np.testing.assert_allclose(lifted.y[i],np.cos(lifted.y[0]),atol=2e-9)
        if var.name=='u_exp': np.testing.assert_allclose(lifted.y[i],np.exp(-lifted.y[0]),atol=2e-9)


def test_composite_trig_argument_uses_total_derivative_and_explicit_input_velocity():
    vf=VarFactory();x,z,out=(vf.add_var(n) for n in ('x','z','out'))
    source=Block(state_vars=[x],diff_vars=[vf.add_diff_var('dx',base_var=x)],state_eqs=[s.Const(2.)],
                 in_vars=[z],algebraic_vars=[out],algebraic_eqs=[out-s.sin(x+z)])
    result=multilinearize_block(source,vf,derivatives={z:s.Const(3.)})
    values=initialize(result,{x.uid:.2,z.uid:.3,out.uid:math.sin(.5)})
    aux=result.children[-1]
    for var,eq in zip(aux.state_vars,aux.state_eqs):
        expected=5*math.cos(.5) if var.name=='u_sin' else -5*math.sin(.5)
        assert eq.eval_uid(values)==pytest.approx(expected)
    assert_multilinear(result)


def test_unsupported_operations_and_missing_kinematics_fail_explicitly():
    vf=VarFactory();x=vf.add_var('x')
    with pytest.raises(ValueError,match='time derivative'):
        multilinearize_block(Block(in_vars=[x],algebraic_eqs=[s.sin(x)]),vf)
    with pytest.raises(ValueError,match='Unsupported multilinear expression'):
        multilinearize_block(Block(in_vars=[x],algebraic_eqs=[s.atan2(x,1)]),vf)
    with pytest.raises(ValueError,match='Unsupported multilinear power'):
        multilinearize_block(Block(in_vars=[x],algebraic_eqs=[x**.3]),vf)


def test_multilinear_block_needs_no_auxiliaries_and_keeps_parameter_programs():
    vf=VarFactory();x,y,p=(vf.add_var(n) for n in ('x','y','p'))
    source=Block(in_vars=[x],algebraic_vars=[y],algebraic_eqs=[y-p*x],event_dict={p:s.sin(s.Const(.2))},
                 init_eqs={y:s.sqrt(x)},out_vars=[y])
    result=multilinearize_block(source,vf)
    assert not result.children
    assert str(result.event_dict[p])==str(source.event_dict[p])
    assert str(result.init_eqs[y])==str(source.init_eqs[y])


def test_shifted_trig_reuses_base_pair_without_extra_states():
    vf=VarFactory();theta=vf.add_var('theta');out=vf.add_var('out')
    expressions=[s.sin(theta),s.cos(theta),s.sin(theta+2*math.pi/3),s.cos(theta-2*math.pi/3),s.sin(.2-theta)]
    source=Block(state_vars=[theta],state_eqs=[s.Const(2.)],
        diff_vars=[vf.add_diff_var('dt',base_var=theta)],algebraic_eqs=expressions)
    result=multilinearize_block(source,vf)
    assert len(result.children[-1].state_vars)==2
    values=initialize(result,{theta.uid:.4})
    for original,converted in zip(expressions,result.algebraic_eqs):
        assert converted.eval_uid(values)==pytest.approx(original.eval_uid({theta.uid:.4}))
    assert_multilinear(result)


def test_standard_generator_runtime_equations_can_be_converted():
    from VeraGridEngine.Templates.Emt.generator_emt_type_template import get_simple_generator_emt_template
    vf=VarFactory()
    source=get_simple_generator_emt_template(vf).block
    result=multilinearize_block(source,vf)
    assert len(result.out_vars)==len(source.out_vars)
    assert [v.uid for v in result.in_vars]==[v.uid for v in source.in_vars]
    assert [v.uid for v in result.out_vars]==[v.uid for v in source.out_vars]
    states=[v for child in result.get_all_blocks() for v in child.state_vars]
    assert len(states)==len(source.state_vars)+2
    # Parameter-only coefficients are allowed by the engine's matrix builder;
    # replace them numerically for the independent monomial inspection.
    parameters={v:s.Const(1.) for child in result.get_all_blocks() for v in list(child.parameters)+list(child.event_dict)}
    for child in result.get_all_blocks():
        for expr in child.state_eqs+child.algebraic_eqs:
            monomials(expr.subs(parameters).simplify())


@pytest.mark.parametrize('seed',[4,19])
def test_standard_generator_residuals_are_preserved_at_arbitrary_points(seed):
    from VeraGridEngine.Templates.Emt.generator_emt_type_template import get_simple_generator_emt_template
    vf=VarFactory()
    source=get_simple_generator_emt_template(vf).block
    result=multilinearize_block(source,vf)
    rng=np.random.default_rng(seed)
    variables={v.uid:v for expr in source.state_eqs+source.algebraic_eqs for v in expr.get_vars()}
    values={uid:float(rng.uniform(.2,1.2)) for uid in variables}
    values=initialize(result,values)
    np.testing.assert_allclose([e.eval_uid(values) for e in source.state_eqs+source.algebraic_eqs],
                               [e.eval_uid(values) for e in result.state_eqs+result.algebraic_eqs],
                               rtol=1e-12,atol=1e-12)
    for child in result.children:
        assert all(abs(eq.eval_uid(values))<1e-12 for eq in child.algebraic_eqs)


def test_logarithm_and_nested_functions_preserve_derivative_chain_rule():
    vf=VarFactory();x=vf.add_var('x')
    source=Block(state_vars=[x],diff_vars=[vf.add_diff_var('dx',base_var=x)],state_eqs=[s.Const(2.)],
                 algebraic_eqs=[s.log(x)+s.exp(s.sin(x))])
    result=multilinearize_block(source,vf)
    values=initialize(result,{x.uid:1.4})
    assert result.algebraic_eqs[0].eval_uid(values)==pytest.approx(math.log(1.4)+math.exp(math.sin(1.4)))
    for child in result.children:
        for var,eq in zip(child.state_vars,child.state_eqs):
            if var.name=='u_log': assert eq.eval_uid(values)==pytest.approx(2/1.4)
            if var.name=='u_exp': assert eq.eval_uid(values)==pytest.approx(2*math.cos(1.4)*math.exp(math.sin(1.4)))
        assert all(abs(eq.eval_uid(values))<1e-12 for eq in child.algebraic_eqs)
    assert_multilinear(result)


@pytest.mark.parametrize('correct_kinematics',[True,False])
def test_preexisting_trig_is_reused_only_when_its_equations_match(correct_kinematics):
    vf=VarFactory();theta,cosine,sine=(vf.add_var(n) for n in ('theta','existing_cos','existing_sin'))
    states=[theta,cosine,sine]
    source=Block(state_vars=states,
        diff_vars=[vf.add_diff_var(f'd_{v.name}',base_var=v) for v in states],
        state_eqs=[s.Const(2.),-2*sine,2*cosine if correct_kinematics else s.Const(0.)],
        init_eqs={cosine:s.cos(theta),sine:s.sin(theta)},algebraic_eqs=[s.sin(theta)+s.cos(theta)])
    result=multilinearize_block(source,vf)
    if correct_kinematics:
        assert not result.children
        assert {v.uid for v in result.algebraic_eqs[0].get_vars()}=={cosine.uid,sine.uid}
    else:
        assert len(result.children[-1].state_vars)==2


def test_unsupported_late_expression_does_not_allocate_partial_factory_lifts():
    vf=VarFactory();x=vf.add_var('x')
    source=Block(state_vars=[x],state_eqs=[s.Const(1.)],
        diff_vars=[vf.add_diff_var('dx',base_var=x)],algebraic_eqs=[s.sin(x),s.atan2(x,1)])
    before_vars,before_diff=len(vf.get_vars_dict()),len(vf.get_diff_var_dict())
    with pytest.raises(ValueError,match='Unsupported'):
        multilinearize_block(source,vf)
    assert (len(vf.get_vars_dict()),len(vf.get_diff_var_dict()))==(before_vars,before_diff)


@pytest.mark.parametrize('value',[-3.,-.7,-.2,.1,.9,2.4])
@pytest.mark.parametrize('form',['hard','minmax','smooth'])
def test_saturations_are_exact_and_repeated_subexpressions_are_shared(value,form):
    from VeraGridEngine.Utils.Symbolic.symbolic_ml import smooth_hard_sat
    vf=VarFactory();x=vf.add_var('x')
    lower,upper=-.7,.9
    def expression():
        if form=='hard':
            return s.hard_sat(x,lower,upper)
        if form=='minmax':
            return s.min(s.max(x,lower),upper)
        return smooth_hard_sat(vf,x,s.Const(lower),s.Const(upper),lam=1e-4)
    original=expression()
    source=Block(in_vars=[x],algebraic_eqs=[original,expression()])
    result=multilinearize_block(source,vf)
    single=multilinearize_block(Block(in_vars=[x],algebraic_eqs=[expression()]),vf)
    count=lambda block: sum(len(b.algebraic_vars)+len(b.state_vars) for b in block.get_all_blocks())
    assert count(result)==count(single)
    values=initialize(result,{x.uid:value})
    if form=='smooth':
        expected=.5*(lower+upper+math.sqrt((value-lower)**2+1e-4)-math.sqrt((value-upper)**2+1e-4))
    else:
        expected=float(np.clip(value,lower,upper))
    for equation in result.algebraic_eqs:
        assert equation.eval_uid(values)==pytest.approx(expected,abs=1e-12)
    for child in result.children:
        assert all(abs(e.eval_uid(values))<1e-12 for e in child.algebraic_eqs)
    assert_multilinear(result)


@pytest.mark.parametrize('value',[-2.,.2,2.])
def test_repeated_exponential_of_saturation_shares_one_exact_dynamic_lift(value):
    vf=VarFactory();x=vf.add_var('x')
    def expression():
        return s.exp(s.hard_sat(x,-.7,.9))
    source=Block(state_vars=[x],state_eqs=[s.Const(2.)],
        diff_vars=[vf.add_diff_var('dx',base_var=x)],algebraic_eqs=[expression()+expression()])
    result=multilinearize_block(source,vf)
    auxiliary_states=[v for b in result.children for v in b.state_vars]
    assert len(auxiliary_states)==1
    assert auxiliary_states[0].name=='u_exp'
    values=initialize(result,{x.uid:value})
    expected=math.exp(float(np.clip(value,-.7,.9)))
    assert result.algebraic_eqs[0].eval_uid(values)==pytest.approx(2*expected)
    aux=result.children[-1]
    expected_derivative=2*expected if -.7<value<.9 else 0.
    assert aux.state_eqs[0].eval_uid(values)==pytest.approx(expected_derivative)
    assert all(abs(e.eval_uid(values))<1e-12 for e in aux.algebraic_eqs)
    assert_multilinear(result)
