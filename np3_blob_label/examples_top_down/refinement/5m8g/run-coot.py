#!/usr/bin/env coot
# python script for coot - generated by dimple
set_nomenclature_errors_on_read("ignore")
import inspect, os
this_file = inspect.getfile(inspect.currentframe())
this_dir = os.path.dirname(os.path.abspath(this_file))
molecule = read_pdb(os.path.join(this_dir, "5m8g.pdb"))
set_rotation_centre(15.61, 20.12, -14.85)
set_zoom(30.)
set_view_quaternion(0.963457, -0.0723914, 0, 0.257897)
refl = os.path.join(this_dir, "5m8g.mtz")
map21 = make_and_draw_map(refl, "FWT", "PHWT", "", 0, 0)
map11 = make_and_draw_map(refl, "DELFWT", "PHDELWT", "", 0, 1)
