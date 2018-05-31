object Util {

  //TODO
  def callUDF(f: Any, in: Seq[Any]): Any = in.size match {
    case 0 => f.asInstanceOf[() => Any]()
    case 1 => f.asInstanceOf[Any => Any](in(0))
    case 2 => f.asInstanceOf[(Any, Any) => Any](in(0), in(1))
    case 3 => f.asInstanceOf[(Any, Any, Any) => Any](in(0), in(1), in(2))
    case _ => ???
  }
}
