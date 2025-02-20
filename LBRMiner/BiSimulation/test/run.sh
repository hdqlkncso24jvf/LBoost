#!/bin/bash

function APair() {
  mpirun -n 4 ../build/her -gd_efile ./business/gd.e \
               -gd_vfile ./business/gd.v \
               -gd_slabel_file ./business/gd_slabels.txt \
               -g_efile ./business/g.e \
               -g_vfile ./business/g.v \
               -g_slabel_file ./business/g_slabels.txt \
               -synonym_file ./business/synonym.txt \
               -embedding_file ./business/glove.6B.300d.txt \
               -bfs_depth 4 \
               -out_prefix ./ \
               -query_type apair
}

function SPair() {
  ../build/her -gd_efile ./business/gd.e \
               -gd_vfile ./business/gd.v \
               -gd_slabel_file ./business/gd_slabels.txt \
               -g_efile ./business/g.e \
               -g_vfile ./business/g.v \
               -g_slabel_file ./business/g_slabels.txt \
               -synonym_file ./business/synonym.txt \
               -embedding_file ./business/glove.6B.300d.txt \
               -bfs_depth 4 \
               -query_type spair \
               -vertex_u 11421 -vertex_v 1470
}

APair
SPair
