
enum class ProducerWarpRole {
      Mainloop = 0,
      Warp1 = 1,
      Epilogue = 2,
      Warp3 = 3
    };

auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
if (warp_group_role == WarpGroupRole::Producer) {
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      // Mainloop Producer Warp
      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        bool do_load_order_arrive = true;
        while (work_tile_info.is_valid()) {
          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          auto k_tile_iter  = cute::make_coord_iterator(shape<3>(gA_mkl));

          collective_mainloop.load(
            params.mainloop,
            mainloop_pipeline,
            mainloop_pipe_producer_state,
            load_inputs,
            blk_coord,
            k_tile_iter, k_tile_count,
            lane_idx,
            block_rank_in_cluster,
            shared_storage.tensors.mainloop
          );
          // Update starting pipeline state for the next tile
          mainloop_pipe_producer_state.advance(k_tile_count);

          // Signal for the epilogue load warp to begin
          if (do_load_order_arrive) {
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }

          // Get next work tile
          scheduler.advance_to_next_work();
          work_tile_info = scheduler.get_current_work();
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);
      } // Mainloop Producer Warp End

      // Epilogue Producer Warp
      else if (producer_warp_role == ProducerWarpRole::Epilogue && collective_epilogue.is_producer_load_needed()) {
        load_order_barrier.wait();
        while (work_tile_info.is_valid()) {
          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          epi_load_pipe_producer_state =
          collective_epilogue.load(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            problem_shape_MNKL,
            blk_shape,
            blk_coord,
            tiled_mma,
            lane_idx,
            shared_storage.tensors.epilogue
          );

          // Get next work tile
          scheduler.advance_to_next_work();
          work_tile_info = scheduler.get_current_work();
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
      } // Epilogue Producer Warp End
    } // Producer Warp Group End
