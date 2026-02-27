import unittest

from analytics import classify_divergence_strength, classify_mci_olsi_divergence


class DivergenceTests(unittest.TestCase):
    def test_strength_buckets_boundaries(self):
        self.assertEqual(classify_divergence_strength(0.09), "NONE")
        self.assertEqual(classify_divergence_strength(0.10), "WEAK")
        self.assertEqual(classify_divergence_strength(0.199), "WEAK")
        self.assertEqual(classify_divergence_strength(0.20), "MODERATE")
        self.assertEqual(classify_divergence_strength(0.35), "MODERATE")
        self.assertEqual(classify_divergence_strength(0.351), "STRONG")

    def test_divergence_payload_for_okx(self):
        div_type, diff, strength, strength_class, mci_norm = classify_mci_olsi_divergence(0.2, 0.4)
        self.assertEqual(mci_norm, 0.6)
        self.assertEqual(diff, 0.2)
        self.assertEqual(strength, 0.2)
        self.assertEqual(div_type, "CALM_WITHOUT_LIQUIDITY")
        self.assertEqual(strength_class, "MODERATE")


if __name__ == "__main__":
    unittest.main()
