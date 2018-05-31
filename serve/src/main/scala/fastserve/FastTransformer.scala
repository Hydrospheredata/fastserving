package fastserve

sealed trait FastTransformer {
  def apply(d: PlainDataset): PlainDataset
  def compose(o: PlainDataset => PlainDataset): FastTransformer
}

object ConstFastTransformer extends FastTransformer {
 def apply(d: PlainDataset): PlainDataset = d
 def compose(o: PlainDataset => PlainDataset): FastTransformer = FastTransformer(o)
}

object FastTransformer {
  def apply(f: PlainDataset => PlainDataset): FastTransformer = new FastTransformer { self =>
    def apply(d: PlainDataset): PlainDataset = f(d)
    def compose(o: PlainDataset => PlainDataset): FastTransformer = FastTransformer((self.apply _ ).compose(o.apply))
  }
}


