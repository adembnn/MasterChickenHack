#!/usr/bin/env python3
"""CPU-only numerical regression checks for isoenergy ablation (synthetic data)."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import t3_t2_tail_isoenergy as exp

class TestIsoenergy(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(20261122)
        n,k=500,100
        x=rng.normal(size=(n,k))
        x-=x.max(1,keepdims=True)
        self.q=np.exp(x)
        self.q/=self.q.sum(1,keepdims=True)
        self.y=rng.integers(k,size=n)
        z=rng.normal(size=(n,k))
        self.teacher=exp.base.conditional_tail(exp.base.at_temperature_two(exp.base.probabilities(z,'logits')),self.y)

    def test_preserve_class_at_both_temperatures(self):
        y=self.y;q=self.q;n=len(q)
        p2=exp.base.at_temperature_two(q)
        for beta in (0.,.2,.4,.6,1.):
            q1,q2,diag=exp.make_target(q,y,self.teacher,beta)
            self.assertTrue(np.isfinite(q1).all())
            self.assertTrue(np.isfinite(q2).all())
            self.assertLess(np.max(abs(q1.sum(1)-1)),1e-10)
            self.assertLess(np.max(abs(q2.sum(1)-1)),1e-10)
            self.assertLess(np.max(abs(q1[np.arange(n),y]-q[np.arange(n),y])),1e-12)
            self.assertLess(np.max(abs(q2[np.arange(n),y]-p2[np.arange(n),y])),5e-6)
            if beta==0.:
                self.assertLess(np.max(abs(q1-q)),1e-12)

    def test_per_image_cross_entropy_unchanged(self):
        q1,_,_=exp.make_target(self.q,self.y,self.teacher,.5)
        rows=np.arange(len(self.y))
        before=-np.log(self.q[rows,self.y]);after=-np.log(q1[rows,self.y])
        self.assertLess(np.max(abs(before-after)),1e-12)
        self.assertLess(abs(before.std()-after.std()),1e-12)

    def test_tail_energy_is_the_T2_y_invariant(self):
        original=exp.base.conditional_tail(exp.base.at_temperature_two(self.q),self.y)
        target_energy=np.square(original).sum(1)
        blended=.5*original+.5*self.teacher
        corrected,supported=exp.project_tail_energy(blended,target_energy)
        self.assertGreater(supported.mean(),.99)
        self.assertLess(np.max(abs(np.square(corrected[supported]).sum(1)-target_energy[supported])),2e-7)

if __name__=='__main__':
    unittest.main()
